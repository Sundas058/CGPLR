from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from cgplr import datasets
from cgplr.diagnostics import correctness, calibration, decision_diagnostics, cluster_size_diagnostics
from cgplr.training import write_json
from cgplr.review_analysis import temporal_transition, deferred_cohort_transition


def diagnose(args):
    run = Path(args.run)
    config = json.loads((run/'config.json').read_text())
    data = datasets.create(config['dataset'], str(Path(args.data_dir)/config['dataset']),
        version=config.get('msmt_version','MSMT17_V1'),combine_trainval=config.get('msmt_combine_trainval',False))
    lookup = {path: int(pid) for path, pid, _ in data.train}
    files = sorted(run.glob('round_*.npz'))
    if not files:
        raise ValueError('no round archives found')
    results = []
    previous = None
    for file in files:
        with np.load(file, allow_pickle=False) as archive:
            names = archive['filenames']
            identities = np.array([lookup[str(name)] for name in names])
            raw, refined = archive['raw_labels'], archive['refined_labels']
            correct, _ = correctness(raw, identities)
            valid = raw >= 0
            result = dict(round=int(file.stem.split('_')[-1]),
                mapping='per-round many-to-one cluster majority; ties choose smallest identity',
                calibration=calibration(archive['scores'][valid], correct[valid], args.bins),
                confidence_bins=calibration(archive['scores'][valid], correct[valid], 5)['bins'],
                decisions=decision_diagnostics(raw, refined, archive['decisions'], identities),
                size_strata=cluster_size_diagnostics(raw,identities,archive['scores'],archive['decisions']),
                cluster_sizes=archive['cluster_sizes'].tolist())
            temporal = archive['temporal_scores']
            result['temporal_summary'] = dict(
                available_history_rounds=min(result['round']-1,config.get('temporal_window',3)),
                mean_valid=float(temporal[valid].mean()) if valid.any() else None,
                mean_correct=float(temporal[valid & correct].mean()) if (valid & correct).any() else None,
                mean_incorrect=float(temporal[valid & ~correct].mean()) if (valid & ~correct).any() else None,
                zero_fraction_correct=float((temporal[valid & correct]==0).mean()) if (valid & correct).any() else None)
            if previous is not None:
                result['temporal_transition'] = temporal_transition(previous,archive,identities)
                result['deferred_cohort_transition'] = deferred_cohort_transition(previous,archive,identities)

            previous = {key:archive[key].copy() for key in
                        ['filenames','raw_labels','decisions','cluster_ids','centroids']}
            results.append(result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'diagnostics.json', results)

    last = results[-min(args.last_rounds, len(results)):]
    summary = {}
    for key in last[0]['decisions']:
        values = [row['decisions'][key] for row in last if row['decisions'][key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    write_json(output/'final_rounds_summary.json', dict(rounds=[r['round'] for r in last], decisions=summary))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with np.load(files[-1], allow_pickle=False) as archive:
            rows = results[-1]['calibration']['bins']
            active = [row for row in rows if row['count']]
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot([0,1], [0,1], '--', color='gray')
            axes[0].plot([r['mean_score'] for r in active], [r['precision'] for r in active], 'o-')
            axes[0].set(xlabel='Mean reliability score', ylabel='Pseudo-label precision', xlim=(0,1), ylim=(0,1))
            margins = archive['margin']
            axes[1].hist(margins[np.isfinite(margins)], bins=30)
            axes[1].set(xlabel='Measured squared-distance margin', ylabel='Samples')
            axes[2].hist(archive['cluster_sizes'], bins=30)
            axes[2].set(xlabel='Raw cluster size', ylabel='Clusters')
            fig.tight_layout()
            fig.savefig(output/'measured_diagnostics.png', dpi=220)
            plt.close(fig)
        cases_file = run/'retrieval_cases.json'
        if cases_file.exists():
            failures = [case for case in json.loads(cases_file.read_text()) if not case['correct_top1']][:5]
            if failures:
                from PIL import Image
                fig, axes = plt.subplots(len(failures),6,figsize=(9,2*len(failures)),squeeze=False)
                for row,case in enumerate(failures):
                    paths = [case['query']]+case['gallery'][:5]
                    for column,axis in enumerate(axes[row]):
                        axis.axis('off')
                        if column < len(paths):
                            with Image.open(paths[column]) as image:
                                axis.imshow(image.convert('RGB'))
                            title = 'Query' if column==0 else ('Correct' if case['matches'][column-1] else 'Incorrect')
                            axis.set_title(title,color='black' if column==0 else ('green' if case['matches'][column-1] else 'red'))
                fig.tight_layout()
                fig.savefig(output/'retrieval_failures.png',dpi=200)
                plt.close(fig)


def aggregate(args):
    from scipy import stats
    runs = [Path(run) for run in args.runs]
    configs = [json.loads((run/'config.json').read_text()) for run in runs]
    if len({config['seed'] for config in configs}) != len(configs):
        raise ValueError('duplicate seeds are not independent runs')
    fixed = {key: value for key, value in configs[0].items() if key not in ('seed','output_dir','checkpoint')}
    for config in configs[1:]:
        if {key: value for key, value in config.items() if key not in ('seed','output_dir','checkpoint')} != fixed:
            raise ValueError('within-method aggregation requires identical settings except seed/path')
    values = [json.loads((run/'metrics.json').read_text()) for run in runs]
    output = dict(n=len(values), seeds=[c['seed'] for c in configs], units='fractions, multiply by 100 for percent')
    for key in ['R1','mAP']:
        samples = np.array([r[key] for r in values])
        output[key] = dict(mean=float(samples.mean()),
                           sample_std=float(samples.std(ddof=1)) if len(samples)>1 else None,
                           values=samples.tolist())
    if args.baselines:
        baselines = {}
        for path in args.baselines:
            config = json.loads((Path(path)/'config.json').read_text())
            if config['seed'] in baselines:
                raise ValueError('duplicate baseline seed')
            baselines[config['seed']] = (Path(path), config)
        if set(baselines) != {c['seed'] for c in configs}:
            raise ValueError('baseline and method seed sets must match exactly')
        protocol_keys = ['dataset','rounds','iters','batch_size','instances','height','width','distance',
                         'eps','min_samples','k1','k2','lr','weight_decay','step_size','lr_gamma',
                         'classifier_temperature','triplet_margin','nondeterministic',
                         'strong_color_jitter','strong_grayscale','strong_erasing','strong_erasing_max',
                         'msmt_version','msmt_combine_trainval']
        base_values = []
        for run, config in zip(runs, configs):
            path, baseline = baselines[config['seed']]
            for key in protocol_keys:
                if baseline.get(key) != config.get(key):
                    raise ValueError('unmatched protocol: '+key)
            a = json.loads((run/'environment.json').read_text())
            b = json.loads((path/'environment.json').read_text())
            if not a.get('checkpoint_sha256') or a.get('checkpoint_sha256') != b.get('checkpoint_sha256'):
                raise ValueError('paired runs must share the source checkpoint for each seed')
            for key in ['torch','torchvision','cuda','gpu','device','numpy','scipy','sklearn']:
                if a.get(key) != b.get(key):
                    raise ValueError('paired environments differ: '+key)
            base_values.append(json.loads((path/'metrics.json').read_text()))
        for key in ['R1','mAP']:
            delta = np.array([a[key]-b[key] for a,b in zip(values,base_values)])
            std = delta.std(ddof=1) if len(delta)>1 else 0
            half = float(stats.t.ppf(0.975,len(delta)-1)*std/np.sqrt(len(delta))) if len(delta)>1 else None
            test = stats.ttest_1samp(delta,0) if len(delta)>1 and std>0 else None
            pvalue = float(test.pvalue) if test is not None else None
            output[key]['paired_comparison'] = dict(mean_difference=float(delta.mean()),
                difference_95_ci=[float(delta.mean()-half),float(delta.mean()+half)] if half is not None else None,
                two_sided_paired_t_p=pvalue,
                caveat='Exploratory, uncorrected for multiple tests; few seeds give weak power. Degenerate variance gives no p-value.')
    write_json(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('diagnose')
    p.add_argument('--run',required=True)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output',required=True)
    p.add_argument('--bins',type=int,default=10)
    p.add_argument('--last-rounds',type=int,default=5)
    p.add_argument('--plot',action='store_true')
    p = commands.add_parser('aggregate')
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--baselines',nargs='+')
    p.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command=='diagnose':
        if args.last_rounds < 1:
            raise ValueError('last-rounds must be positive')
        diagnose(args)
    else:
        aggregate(args)


if __name__=='__main__':
    main()
