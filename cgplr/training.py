import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import datasets
from .models.encoder import ReIDEncoder
from .training_data import Images, SelectiveBatchSampler, transforms
from .method import (ClusterHistory, estimate_sample_confidence, adaptive_selective_refinement,
                     compact_cluster_labels, compute_cluster_centroids, ReliabilityAwareSelectiveLoss)
from .method.clustering import cluster_with_dbscan
from .method.source_training import source_reid_loss
from .diagnostics import retrieval_metrics
from .experiments import ABLATIONS, apply_ablation, deferred_batch_mask
from .reproducibility import ProcessMemory, source_fingerprints


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def measured(timings, key, device):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    timings[key] = timings.get(key, 0.0) + time.perf_counter()-start


def loader(records, transform, args, batch_sampler=None, strong=None, seed_offset=0):
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    common = dict(num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                  worker_init_fn=seed_worker, generator=generator)
    data = Images(records, transform, strong)
    if batch_sampler is not None:
        return DataLoader(data, batch_sampler=batch_sampler, **common)
    return DataLoader(data, batch_size=args.batch_size, shuffle=False, **common)


@torch.no_grad()
def extract(model, records, args, device):
    model.eval()
    features = []
    for images, _ in loader(records, transforms(args.height, args.width, training=False), args):
        features.append(model(images.to(device)).cpu())
    if not features:
        raise ValueError('empty feature extraction dataset')
    return torch.cat(features)


def save_model(path, model, args, kind, extra=None):
    state = dict(encoder=model.state_dict(), kind=kind, config=vars(args), extra=extra or {})
    torch.save(state, path)


def load_model(model, checkpoint, expected=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if expected and state.get('kind') != expected:
        raise ValueError('Expected a canonical '+expected+' checkpoint; legacy checkpoints require explicit conversion')
    model.load_state_dict(state['encoder'], strict=True)
    return state


def optimizer(parameters, args):
    return torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)


def train_source(model, dataset, args, output, device):
    started = time.perf_counter()
    if args.source_epochs is None or args.source_epochs < 1:
        raise ValueError('--source-epochs must explicitly state Ts')
    rows = sorted(dataset.train)
    identities = sorted({row[1] for row in rows})
    remap = {pid: i for i, pid in enumerate(identities)}
    labels = torch.tensor([remap[row[1]] for row in rows], device=device)
    head = nn.Linear(model.num_features, len(identities), bias=False).to(device)
    opt = optimizer(list(model.parameters())+list(head.parameters()), args)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.step_size, gamma=args.lr_gamma)
    timings = {}
    epoch_logs = []
    for epoch in range(args.source_epochs):
        sampler = SelectiveBatchSampler(labels.cpu(), [False]*len(rows), args.batch_size,
                                         args.instances, args.iters, deferred_fraction=0)
        batches = loader(rows, transforms(args.height, args.width), args, sampler, seed_offset=epoch+1)
        model.train()
        total = 0.0
        with measured(timings, 'source_optimization_seconds', device):
            for images, indices in batches:
                features = model(images.to(device))
                logits = F.linear(features, F.normalize(head.weight, dim=1))/args.classifier_temperature
                loss = source_reid_loss(logits, features, labels[indices.to(device)],
                                        triplet_margin=args.triplet_margin, beta=args.source_triplet_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
        scheduler.step()
        epoch_logs.append(dict(source_epoch=epoch+1, loss=total/len(batches),
                               next_lr=opt.param_groups[0]['lr']))
        write_json(output/'source_epochs.json',epoch_logs)
        print(json.dumps(epoch_logs[-1]), flush=True)
    timings['source_wall_seconds'] = time.perf_counter()-started
    save_model(output/'source.pth', model, args, 'source', timings)
    return timings


def cluster_features(features, args):
    if args.distance == 'euclidean':

        from sklearn.neighbors import radius_neighbors_graph
        from sklearn.neighbors import sort_graph_by_row_values
        from sklearn.cluster import DBSCAN
        graph = radius_neighbors_graph(features.numpy(), args.eps, mode='distance',
                                       include_self=True, n_jobs=args.cluster_jobs)
        graph = sort_graph_by_row_values(graph, warn_when_not_sorted=False)
        return DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed',
                      n_jobs=args.cluster_jobs).fit_predict(graph)

    from .utils.faiss_rerank import compute_jaccard_distance
    if len(features) <= max(args.k1, args.k2):
        raise ValueError('Jaccard k1/k2 must be smaller than the target sample count')
    distance = compute_jaccard_distance(features, k1=args.k1, k2=args.k2, search_option=3)
    distance = np.maximum((distance+distance.T)*0.5, 0)
    np.fill_diagonal(distance, 0)
    return cluster_with_dbscan(distance, args.eps, args.min_samples, args.cluster_jobs)


def adapt(model, dataset, args, output, device):

    rows = [(path, -1, cam) for path, _, cam in sorted(dataset.train)]
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('duplicate training filenames prevent stable history indexing')
    order_hash = hashlib.sha256('\n'.join(row[0] for row in rows).encode()).hexdigest()
    history = ClusterHistory(args.temporal_window)
    opt = optimizer(model.parameters(), args)
    schedule = torch.optim.lr_scheduler.StepLR(opt, args.step_size, gamma=args.lr_gamma)
    criterion = ReliabilityAwareSelectiveLoss(args.triplet_margin, args.triplet_weight,
                                             args.deferred_weight, args.numerical_eps)
    timings, rounds = {}, []
    initial = time.perf_counter()
    for round_index in range(args.rounds):
        row = dict(round=round_index+1)
        with measured(row, 'feature_extraction_seconds', device):
            features = extract(model, rows, args, device)
        with measured(row, 'clustering_seconds', device):
            raw = cluster_features(features, args)
        if args.clustering_baseline:
            raw_tensor = torch.tensor(raw).long()
            raw_ids, raw_centers = compute_cluster_centroids(features, raw_tensor)
            confidence = SimpleNamespace(weights=(raw_tensor>=0).float(), cluster_ids=raw_ids,
                centroids=raw_centers, temporal_scores=torch.zeros(len(raw)),
                assigned_distances=torch.zeros(len(raw)), competitor_distances=torch.full((len(raw),),torch.nan))
            result = SimpleNamespace(refined_labels=raw_tensor,
                decisions=torch.where(raw_tensor>=0,1,0), learning_weights=(raw_tensor>=0).float())
        else:
            with measured(row, 'reliability_seconds', device):
                confidence = estimate_sample_confidence(features, raw, alpha=args.alpha,
                    centroid_temperature=args.centroid_temperature, margin_temperature=args.margin_temperature,
                    neighbors=args.neighbors, fusion_weights=(args.structural_weight, args.margin_weight,
                    args.temporal_weight), history=history, return_details=True, temporal_mode=args.temporal_mode)
        with measured(row, 'refinement_seconds', device):
            if not args.clustering_baseline and args.refinement_mode == 'raw':
                raw_tensor = torch.tensor(raw).long()
                result = SimpleNamespace(refined_labels=raw_tensor,
                    decisions=torch.where(raw_tensor >= 0, 1, 0), learning_weights=confidence.weights.clone())
            elif not args.clustering_baseline:
                result = adaptive_selective_refinement(features, raw, confidence, rho_high=args.rho_high,
                    rho_low=args.rho_low, dispersion_eps=args.dispersion_eps, reassignment_beta=args.beta,
                    reassignment_temperature=args.reassignment_temperature, evidence_gain=args.evidence_gain,
                    neighbor_support=args.neighbor_support, reassigned_weight=args.reassigned_weight,
                    neighbors=args.neighbors, eps=args.numerical_eps, min_candidate_size=args.min_candidate_size,
                    min_candidate_reliability=args.min_candidate_reliability,
                    global_thresholds=(args.global_low, args.global_high) if args.global_thresholds else None,
                    force_reassignment=args.force_reassignment)
            if args.uniform_weights:
                result.learning_weights = torch.isin(result.decisions, torch.tensor([1, 2])).float()
            labels, _ = compact_cluster_labels(result.refined_labels)
            ids, centers = compute_cluster_centroids(features, labels)
        row['clusters'] = len(ids)
        row['outliers'] = int((torch.tensor(raw) < 0).sum())
        for value, name in [(1, 'preserved'), (2, 'reassigned'), (3, 'deferred')]:
            row[name] = int((result.decisions == value).sum())
        _, sizes = np.unique(np.asarray(raw)[np.asarray(raw) >= 0], return_counts=True)
        row['cluster_sizes'] = dict(count=int(len(sizes)), minimum=int(sizes.min()) if len(sizes) else None,
                                    median=float(np.median(sizes)) if len(sizes) else None,
                                    maximum=int(sizes.max()) if len(sizes) else None)
        row['sample_order_sha256'] = order_hash
        archive = dict(
            filenames=np.array([r[0] for r in rows]), raw_labels=raw,
            refined_labels=result.refined_labels.cpu().numpy(), decisions=result.decisions.cpu().numpy(),
            scores=confidence.weights.cpu().numpy(), learning_weights=result.learning_weights.cpu().numpy(),
            margin=(confidence.competitor_distances-confidence.assigned_distances).cpu().numpy(),
            temporal_scores=confidence.temporal_scores.cpu().numpy(), cluster_sizes=sizes,
            cluster_ids=confidence.cluster_ids.cpu().numpy(), centroids=confidence.centroids.cpu().numpy())
        if not args.clustering_baseline:
            for name in ['centroid_scores','neighborhood_scores','structural_scores','margin_scores']:
                archive[name] = getattr(confidence, name).cpu().numpy()
        for name in ['upper_thresholds','lower_thresholds','best_alternatives','evidence_gain','alternative_support']:
            if hasattr(result,name):
                archive[name] = getattr(result,name).cpu().numpy()
        if args.save_features:
            archive['features'] = features.cpu().numpy()
        np.savez_compressed(output/f'round_{round_index+1:03d}.npz', **archive)
        deferred = result.decisions == 3
        active_deferred = deferred_batch_mask(deferred, args.discard_deferred)
        row['deferred_in_batches'] = bool(active_deferred.any())
        row['consistency_enabled'] = bool(active_deferred.any()) and args.deferred_weight > 0
        if len(ids) or active_deferred.any():
            head = nn.Linear(model.num_features, max(len(ids), 1), bias=False).to(device)
            if len(ids):
                with torch.no_grad():
                    head.weight.copy_(centers.to(device))
            head_opt = optimizer(head.parameters(), args)
            for group in head_opt.param_groups:
                group['lr'] = opt.param_groups[0]['lr']
            sampler = SelectiveBatchSampler(labels, active_deferred, args.batch_size, args.instances,
                                            args.iters, args.deferred_fraction if active_deferred.any() else 0)
            batches = loader(rows, transforms(args.height, args.width), args, sampler,
                              strong=transforms(args.height, args.width, strong=True,
                                  color_jitter=args.strong_color_jitter, grayscale=args.strong_grayscale,
                                  erasing=args.strong_erasing, erasing_max=args.strong_erasing_max),
                              seed_offset=1000+round_index)
            model.train()
            total = 0.0
            updates = 0
            labels_device = labels.to(device)
            decisions = result.decisions.to(device)
            weights = result.learning_weights.to(device)
            with measured(row, 'optimization_seconds', device):
                for weak, strong, indices in batches:
                    indices = indices.to(device)
                    weak_features = model(weak.to(device))
                    logits = F.linear(weak_features, F.normalize(head.weight, dim=1))/args.classifier_temperature

                    mask = decisions[indices] == 3
                    if mask.any() and args.deferred_weight > 0:
                        strong_features = model(strong.to(device))[mask]
                        from .method.objectives import deferred_consistency_loss
                        consistency = deferred_consistency_loss(weak_features[mask], strong_features,
                                                                eps=args.numerical_eps)
                    else:
                        consistency = weak_features.sum()*0
                    loss = criterion(logits, weak_features, labels_device[indices], weights[indices],
                                     decisions[indices]) + args.deferred_weight*consistency
                    opt.zero_grad(set_to_none=True)
                    head_opt.zero_grad(set_to_none=True)
                    has_signal = bool(torch.isin(decisions[indices], torch.tensor([1,2],device=device)).any()) or (bool(mask.any()) and args.deferred_weight > 0)
                    if has_signal:
                        loss.backward()
                        opt.step()
                        head_opt.step()
                        updates += 1
                    total += float(loss.detach())
            row['mean_loss'] = total/len(batches)
            row['optimizer_updates'] = updates
            if updates:
                schedule.step()
        else:
            row['skipped_no_learning_samples'] = True

        history.append(raw, confidence.cluster_ids, confidence.centroids)
        rounds.append(row)
        write_json(output/'rounds.json', rounds)
        print(json.dumps(row), flush=True)
    timings['adaptation_wall_seconds'] = time.perf_counter()-initial
    for key in ('feature_extraction_seconds', 'clustering_seconds', 'reliability_seconds',
                'refinement_seconds', 'optimization_seconds'):
        timings[key] = sum(row.get(key, 0.0) for row in rounds)
    save_model(output/'final.pth', model, args, 'adapted', timings)
    return timings


def evaluate(model, dataset, args, output, device):
    model.eval()
    timings = {}
    with measured(timings, 'query_extraction_seconds', device):
        qfeatures = extract(model, dataset.query, args, device)
    with measured(timings, 'gallery_extraction_seconds', device):
        gfeatures = extract(model, dataset.gallery, args, device)
    with measured(timings, 'gallery_ranking_seconds', device):
        metrics, cases = retrieval_metrics(qfeatures, gfeatures, dataset.query, dataset.gallery)
    if args.save_retrieval_features:
        np.savez_compressed(output/'retrieval_features.npz', query_features=qfeatures.numpy(),
            gallery_features=gfeatures.numpy(), query_filenames=np.array([r[0] for r in dataset.query]),
            gallery_filenames=np.array([r[0] for r in dataset.gallery]),
            query_ids=np.array([r[1] for r in dataset.query]), gallery_ids=np.array([r[1] for r in dataset.gallery]),
            query_cameras=np.array([r[2] for r in dataset.query]), gallery_cameras=np.array([r[2] for r in dataset.gallery]))

    batch = torch.zeros(args.benchmark_batch, 3, args.height, args.width, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(batch)
        with measured(timings, 'benchmark_forward_seconds', device):
            for _ in range(args.benchmark_repeats):
                model(batch)
    seconds = timings['benchmark_forward_seconds']/args.benchmark_repeats
    metrics.update(timings)
    breakdown = {name: sum(p.numel() for p in module.parameters())
                 for name, module in model.named_children()}
    flop_count = [0]
    def count_flops(module, inputs, result):
        if isinstance(module, nn.Conv2d):
            flop_count[0] += int(result.numel()*2*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            flop_count[0] += int(result.numel()*2*module.in_features)
    handles = [module.register_forward_hook(count_flops) for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            model(batch[:1])
    finally:
        for handle in handles:
            handle.remove()
    metrics.update(benchmark_batch=args.benchmark_batch, benchmark_warmup=10,
                   benchmark_repeats=args.benchmark_repeats,precision='float32',
                   input_height=args.height,input_width=args.width,forward_batch_ms=seconds*1000,
                   amortized_image_ms=seconds*1000/args.benchmark_batch,
                   encoder_fps=args.benchmark_batch/seconds,
                   encoder_parameters=sum(p.numel() for p in model.parameters()),
                   parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                   parameter_breakdown=breakdown, conv_linear_flops_per_image=flop_count[0],
                   flops_convention='2 operations per multiply-add; convolution and linear only')
    write_json(output/'metrics.json', metrics)
    write_json(output/'retrieval_cases.json', cases)
    print(json.dumps(metrics), flush=True)
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['source', 'adapt', 'evaluate'], default='adapt')
    p.add_argument('--config', help='JSON defaults; explicit CLI values override config values')
    p.add_argument('--ablation', choices=sorted(ABLATIONS), default='full')
    p.add_argument('--refinement-mode', choices=['selective','raw'], default='selective')
    p.add_argument('--dataset', required=True, choices=datasets.names())
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--imagenet', action='store_true', help='initialize source stage from torchvision ImageNet V1')
    p.add_argument('--source-epochs', type=int, help='number of source-training epochs')
    p.add_argument('--source-triplet-weight', type=float, default=1.0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--nondeterministic', action='store_true')
    for name, default in [('batch-size',64), ('instances',4), ('iters',200), ('rounds',50), ('workers',4),
                          ('height',256), ('width',128), ('step-size',20), ('neighbors',5),
                          ('temporal-window',3), ('min-samples',4), ('cluster-jobs',-1), ('k1',30), ('k2',6),
                          ('min-candidate-size',1), ('benchmark-batch',1), ('benchmark-repeats',50)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',3.5e-4), ('weight-decay',5e-4), ('lr-gamma',0.1), ('triplet-margin',0.3),
            ('alpha',0.5), ('beta',0.5), ('centroid-temperature',0.5), ('margin-temperature',0.1),
            ('reassignment-temperature',0.5), ('structural-weight',0.5), ('margin-weight',0.3),
            ('temporal-weight',0.2), ('rho-high',1.0), ('rho-low',1.0), ('evidence-gain',0.1),
            ('neighbor-support',0.6), ('reassigned-weight',0.7), ('dispersion-eps',1e-6),
            ('numerical-eps',1e-12), ('triplet-weight',1.0), ('deferred-weight',0.1),
            ('deferred-fraction',0.25), ('eps',0.55), ('classifier-temperature',0.05),
            ('min-candidate-reliability',0.0), ('global-low',0.4), ('global-high',0.6)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--strong-color-jitter',type=float,default=0.2)
    p.add_argument('--strong-grayscale',type=float,default=0.1)
    p.add_argument('--strong-erasing',type=float,default=0.5)
    p.add_argument('--strong-erasing-max',type=float,default=0.2)
    p.add_argument('--distance', choices=['jaccard','euclidean'], default='jaccard')
    p.add_argument('--temporal-mode', choices=['centroid','overlap'], default='centroid')
    p.add_argument('--save-features',action='store_true')
    p.add_argument('--save-retrieval-features',action='store_true')
    p.add_argument('--msmt-version',choices=['MSMT17_V1','MSMT17_V2'],default='MSMT17_V1')
    p.add_argument('--msmt-combine-trainval',action='store_true')
    for name in ['clustering-baseline', 'global-thresholds', 'force-reassignment', 'uniform-weights', 'discard-deferred']:
        p.add_argument('--'+name, action='store_true')
    return p


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config')
    known, _ = bootstrap.parse_known_args(argv)
    p = parser()
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding='utf-8-sig'))
        if not isinstance(defaults,dict):
            raise ValueError('config must be a JSON object')
        allowed = {action.dest for action in p._actions} - {'help','config'}
        if set(defaults)-allowed:
            raise ValueError('unknown config fields: '+','.join(sorted(set(defaults)-allowed)))

        configured = []
        for action in p._actions:
            if action.dest not in defaults:
                continue
            value = defaults[action.dest]
            if isinstance(action,argparse._StoreTrueAction):
                if not isinstance(value,bool):
                    raise ValueError(action.dest+' requires a JSON boolean')
                if value: configured.append(action.option_strings[-1])
            elif value is not None:
                configured.extend([action.option_strings[-1],str(value)])
        import sys
        parsed = p.parse_args(configured + (list(argv) if argv is not None else sys.argv[1:]))
    else:
        parsed = p.parse_args(argv)
    return apply_ablation(parsed)


def main(argv=None):
    args = parse_args(argv)
    if any(isinstance(value,float) and not np.isfinite(value) for value in vars(args).values()):
        raise ValueError('all numerical settings must be finite')
    for name in ['lr','classifier_temperature','centroid_temperature','margin_temperature',
                 'reassignment_temperature','dispersion_eps','numerical_eps','eps']:
        if getattr(args,name) <= 0:
            raise ValueError(name+' must be positive')
    for name in ['alpha','beta','neighbor_support','deferred_fraction','min_candidate_reliability',
                 'strong_color_jitter','strong_grayscale','strong_erasing']:
        if not 0 <= getattr(args,name) <= 1:
            raise ValueError(name+' must lie in [0,1]')
    for name in ['structural_weight','margin_weight','temporal_weight','triplet_weight','deferred_weight',
                 'source_triplet_weight','rho_high','rho_low','evidence_gain','weight_decay']:
        if getattr(args,name) < 0:
            raise ValueError(name+' must be nonnegative')
    if args.structural_weight+args.margin_weight <= 0:
        raise ValueError('structural and margin weights cannot both be zero')
    if not 0 < args.reassigned_weight <= 1 or args.instances < 2 or args.temporal_window < 0:
        raise ValueError('invalid reassigned weight, instances or temporal window')
    if not 0.02 <= args.strong_erasing_max <= 1:
        raise ValueError('strong-erasing-max must lie in [0.02,1]')
    if args.stage=='source' and (args.source_epochs is None or args.source_epochs < 1):
        raise ValueError('--source-epochs explicitly states Ts and is required for source training')
    for name in ['rounds', 'iters', 'batch_size', 'instances', 'height', 'width', 'step_size',
                 'benchmark_batch', 'benchmark_repeats', 'neighbors', 'min_samples']:
        if getattr(args, name) < 1:
            raise ValueError(name+' must be positive')
    if args.batch_size < 2 or args.workers < 0:
        raise ValueError('batch size >=2 and workers >=0 required')
    if args.stage in ('adapt','evaluate') and not args.checkpoint:
        raise ValueError('--checkpoint is required for adaptation/evaluation')
    if args.clustering_baseline:
        args.deferred_weight = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists():
        raise FileExistsError('use a new output directory for each run; refusing to overwrite evidence')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = not args.nondeterministic
    torch.backends.cudnn.benchmark = args.nondeterministic
    torch.use_deterministic_algorithms(not args.nondeterministic)
    device = torch.device(args.device)
    if device.type == 'cuda':

        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output/'config.json', vars(args))
    import torchvision, scipy, sklearn
    environment = dict(python=platform.python_version(), torch=str(torch.__version__),
                       torchvision=str(torchvision.__version__), scipy=scipy.__version__, sklearn=sklearn.__version__,
                       numpy=np.__version__, cuda=torch.version.cuda, device=str(device),
                       gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    if args.checkpoint:
        with open(args.checkpoint, 'rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        environment['checkpoint_sha256'] = digest
    write_json(output/'environment.json', environment)
    start = time.perf_counter()
    dataset = datasets.create(args.dataset, str(Path(args.data_dir)/args.dataset),
                               version=args.msmt_version, combine_trainval=args.msmt_combine_trainval)
    write_json(output/'dataset_manifest.json', dict(dataset=args.dataset,
        train=[dict(filename=r[0],camera=int(r[2])) for r in sorted(dataset.train)],
        query_filenames=[r[0] for r in dataset.query],gallery_filenames=[r[0] for r in dataset.gallery],
        train_count=len(dataset.train),query_count=len(dataset.query),gallery_count=len(dataset.gallery)))
    write_json(output/'source_fingerprints.json',source_fingerprints(Path(__file__).resolve().parents[1]))
    model = ReIDEncoder(imagenet=args.imagenet and args.stage=='source').to(device)
    checkpoint_state = None
    if args.checkpoint:
        checkpoint_state = load_model(model, args.checkpoint, 'source' if args.stage=='adapt' else None)
        write_json(output/'initialization_config.json',checkpoint_state.get('config',{}))
    with ProcessMemory() as memory:
        if args.stage == 'source':
            timings = train_source(model, dataset, args, output, device)
        elif args.stage == 'adapt':
            timings = adapt(model, dataset, args, output, device)
        else:
            timings = {}
    timings['sampled_process_rss_peak_bytes'] = memory.peak
    timings['rss_sample_interval_seconds'] = memory.interval
    timings['peak_training_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(device)
        if device.type=='cuda' and args.stage!='evaluate' else None)
    if args.stage != 'source':
        evaluate(model, dataset, args, output, device)
    timings['stage_wall_seconds'] = time.perf_counter()-start
    if checkpoint_state and args.stage == 'adapt':
        timings['source_optimization_seconds_from_checkpoint'] = checkpoint_state.get('extra', {}).get('source_optimization_seconds')
        source_wall = checkpoint_state.get('extra', {}).get('source_wall_seconds')
        timings['source_wall_seconds_from_checkpoint'] = source_wall
        timings['source_plus_adaptation_seconds'] = (source_wall+timings['adaptation_wall_seconds']) if source_wall is not None else None
    timings['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    timings['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
    write_json(output/'timing.json', timings)


if __name__ == '__main__':
    main()
