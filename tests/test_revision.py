import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from cgplr.method import (estimate_sample_confidence, adaptive_selective_refinement, ClusterHistory,
    confidence_weighted_triplet_loss, deferred_consistency_loss, ReliabilityAwareSelectiveLoss)
from cgplr.method.history import minimum_cost_centroid_alignment, temporal_assignment_stability
from cgplr.method.neighbors import nearest_neighbors
from cgplr.diagnostics import calibration, decision_diagnostics, retrieval_metrics
from cgplr.training_data import SelectiveBatchSampler


class RevisionTests(unittest.TestCase):
    def test_exact_matching_beats_greedy(self):
        mapping = minimum_cost_centroid_alignment([0,1], [[0.],[2.]], [10,11], [[1.1],[3.]])
        self.assertEqual(mapping, {0:10,1:11})

    def test_temporal_permutation_and_order_guard(self):
        history = ClusterHistory(3)
        history.append([0,0,1,1], [0,1], [[1.,0.],[0.,1.]])
        score = temporal_assignment_stability([7,7,9,9],[7,9],[[1.,0.],[0.,1.]],history)
        torch.testing.assert_close(score, torch.ones(4))
        with self.assertRaises(ValueError):
            temporal_assignment_stability([7,9],[7,9],[[1.,0.],[0.,1.]],history)

    def test_overlap_split_explicit_ablation(self):
        history = ClusterHistory(1)
        history.append([0,0,0,0],[0],[[1.,0.]])
        score = temporal_assignment_stability([1,1,2,2],[1,2],[[1.,0.],[0.,1.]],history,mode='overlap')
        torch.testing.assert_close(score, torch.full((4,),0.5))

    def test_neighbors_never_include_self_even_antipodal(self):
        neighbors = nearest_neighbors(torch.tensor([[1.,0.],[-1.,0.]]),1,chunk_size=1)
        self.assertEqual(neighbors.tolist(), [[1],[0]])

    def test_single_valid_sample_has_centroid_evidence(self):
        result = estimate_sample_confidence([[1.,0.],[0.,1.]],[0,-1],return_details=True)
        self.assertEqual(result.cluster_ids.tolist(),[0])
        self.assertEqual(result.centroid_scores.tolist(),[1.,0.])
        self.assertEqual(result.margin_scores.tolist(),[0.5,0.])
        self.assertTrue(torch.isfinite(result.weights).all())

    def test_active_cue_renormalization(self):
        result = estimate_sample_confidence([[1.,0.],[0.,1.]],[0,1],fusion_weights=(0.5,0,0.2),return_details=True)
        torch.testing.assert_close(result.weights,result.structural_scores)

    def test_all_outliers_remain_outside_partition(self):
        features = torch.eye(2)
        detail = estimate_sample_confidence(features,[-1,-1],return_details=True)
        result = adaptive_selective_refinement(features,[-1,-1],detail)
        self.assertEqual(result.decisions.tolist(),[0,0])
        self.assertEqual(result.refined_labels.tolist(),[-1,-1])

    def test_reassignment_checks_neighbor_votes_not_fused_evidence(self):
        features = torch.tensor([[1.,0.],[1.,0.],[-1.,0.],[0.99,0.1],[0.99,-0.1],[1.,0.2]])
        labels = torch.tensor([0,0,0,1,1,1])
        detail = estimate_sample_confidence(features,labels,return_details=True)
        detail.centroids = torch.tensor([[-1.,0.],[1.,0.]])
        detail.weights = torch.tensor([0.5,0.4,0.6,0.9,0.9,0.9])
        result = adaptive_selective_refinement(features,labels,detail,neighbors=1,reassignment_beta=1,
                                              neighbor_support=0.6)
        self.assertEqual(int(result.decisions[0]),3)
        self.assertEqual(float(result.alternative_support[0]),0)
        self.assertGreater(float(result.evidence_gain[0]),0.6)
        forced = adaptive_selective_refinement(features,labels,detail,neighbors=1,reassignment_beta=1,
                                               neighbor_support=0.6,force_reassignment=True)
        self.assertEqual(int(forced.decisions[0]),2)
        self.assertGreater(float(forced.learning_weights[0]),0.6)
        guarded = adaptive_selective_refinement(features,labels,detail,neighbors=1,reassignment_beta=1,
                                               force_reassignment=True,min_candidate_size=4)
        self.assertEqual(int(guarded.decisions[0]),3)

    def test_triplet_uses_squared_distance_and_eligible_anchors(self):
        features = torch.tensor([[1.,0.],[0.,1.],[0.8,0.6]],requires_grad=True)
        labels = torch.tensor([0,0,1])
        weights = torch.tensor([0.2,0.8,1.0],requires_grad=True)
        loss = confidence_weighted_triplet_loss(features,labels,weights,margin=0.3)

        torch.testing.assert_close(loss,torch.tensor(1.58))
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_empty_and_deferred_losses_backpropagate(self):
        features = torch.randn(3,4,requires_grad=True)
        logits = torch.randn(3,2,requires_grad=True)
        loss = ReliabilityAwareSelectiveLoss()(logits,features,torch.full((3,),-1),torch.zeros(3),torch.full((3,),3))
        loss.backward()
        self.assertEqual(float(loss.detach()),0)
        weak = torch.randn(3,4,requires_grad=True)
        strong = torch.randn(3,4,requires_grad=True)
        loss = deferred_consistency_loss(weak,strong)
        loss.backward()
        self.assertIsNone(weak.grad)
        self.assertIsNotNone(strong.grad)
        self.assertTrue(0 <= float(loss.detach()) <= 4)

    def test_sampler_excludes_outliers_supports_deferred_only(self):
        sampler = SelectiveBatchSampler([-1,-1,-1],[False,True,True],4,2,2)
        self.assertTrue(all(len(batch)==4 and 0 not in batch for batch in sampler))

    def test_calibration_and_empty_conditionals(self):
        result = calibration([0.,1.],[False,True])
        self.assertEqual(result['ece'],0)
        self.assertEqual(sum(row['count'] for row in result['bins']),2)
        metrics = decision_diagnostics([0,0],[0,0],[1,1],[10,10])
        self.assertIsNone(metrics['correction_rate'])
        self.assertIsNone(metrics['deferred_original_error'])

    def test_same_camera_exclusion(self):
        query = [('q',1,0)]
        gallery = [('same-camera',1,0),('wrong',2,1),('match',1,1)]
        metrics, cases = retrieval_metrics([[1.,0.]], [[1.,0.],[0.9,0.1],[0.,1.]],query,gallery)
        self.assertEqual(metrics['R1'],0)
        self.assertEqual(metrics['mAP'],0.5)
        self.assertEqual(cases[0]['gallery'][0],'wrong')

    def test_paired_summary_rejects_unmatched_protocol(self):
        from examples.analyze_revision import aggregate
        from cgplr.training import parser, write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, baselines = [], []
            for seed in [1,2,3]:
                for method in ['full','baseline']:
                    path = root/f'{method}_{seed}'
                    path.mkdir()
                    config = vars(parser().parse_args(['--dataset','market1501','--output-dir',str(path),
                        '--checkpoint','source.pth','--seed',str(seed)]))
                    config['clustering_baseline'] = method=='baseline'
                    write_json(path/'config.json',config)
                    write_json(path/'environment.json',dict(checkpoint_sha256='test-source-checkpoint'))
                    write_json(path/'metrics.json',dict(R1=0.6+0.01*seed if method=='full' else 0.5,
                                                       mAP=0.4+0.01*seed if method=='full' else 0.3))
                    (runs if method=='full' else baselines).append(str(path))
            args = SimpleNamespace(runs=runs,baselines=baselines,output=str(root/'summary.json'))
            aggregate(args)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['n'],3)
            self.assertGreater(summary['mAP']['sample_std'],0)
            self.assertIsNotNone(summary['mAP']['paired_comparison']['two_sided_paired_t_p'])
            bad = Path(baselines[0])/'config.json'
            config = json.loads(bad.read_text())
            config['rounds'] += 1
            write_json(bad,config)
            with self.assertRaises(ValueError):
                aggregate(args)

    def test_tiny_source_adaptation_and_posthoc_evaluation(self):
        from cgplr.training import parser, train_source, adapt, evaluate, load_model
        torch.set_num_threads(1)
        class TinyEncoder(nn.Module):
            num_features = 4
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3,4,1)
            def forward(self,x):
                return F.normalize(self.conv(x).mean((2,3)),dim=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(8):
                path = root/f'{index}.png'
                rng = np.random.default_rng(index)
                Image.fromarray(rng.integers(0,255,(32,16,3),dtype=np.uint8)).save(path)
                rows.append((str(path),index//4,index%2))
            data = SimpleNamespace(train=rows,query=[rows[0],rows[4]],gallery=[rows[1],rows[5]])
            args = parser().parse_args(['--dataset','market1501','--output-dir',str(root),
                '--device','cpu','--source-epochs','1','--rounds','2','--iters','1','--workers','0',
                '--batch-size','4','--instances','2','--height','32','--width','16',
                '--distance','euclidean','--eps','2.1','--min-samples','2',
                '--benchmark-repeats','2'])
            args.save_features = True
            args.save_retrieval_features = True
            model = TinyEncoder()
            device = torch.device('cpu')
            train_source(model,data,args,root,device)
            load_model(TinyEncoder(),root/'source.pth','source')
            adapt(model,data,args,root,device)
            metrics = evaluate(model,data,args,root,device)
            self.assertTrue((root/'final.pth').is_file())
            self.assertEqual(len(json.loads((root/'rounds.json').read_text())),2)
            self.assertTrue(0 <= metrics['mAP'] <= 1)
            with np.load(root/'round_002.npz') as archive:
                self.assertNotIn('identities',archive.files)
                for key in ['features','centroids','centroid_scores','neighborhood_scores',
                            'margin_scores','upper_thresholds','evidence_gain','alternative_support']:
                    self.assertIn(key,archive.files)
                self.assertEqual(len(archive['features']),len(rows))
            self.assertTrue((root/'retrieval_features.npz').exists())

            only_deferred = root/'only_deferred'
            only_deferred.mkdir()
            args.rho_high = 1e8
            args.rounds = 1
            adapt(model,data,args,only_deferred,device)
            round_info = json.loads((only_deferred/'rounds.json').read_text())[0]
            self.assertEqual(round_info['clusters'],0)
            self.assertEqual(round_info['deferred'],len(rows))
            self.assertIn('optimization_seconds',round_info)


if __name__=='__main__':
    unittest.main()
