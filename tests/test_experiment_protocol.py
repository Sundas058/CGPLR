import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from cgplr.training import parse_args
from cgplr.experiments import deferred_batch_mask
from cgplr.diagnostics import cluster_size_diagnostics
from cgplr.datasets.msmt17 import MSMT17
from examples.run_experiments import build_runs
from PIL import Image
from torch import nn
from unittest.mock import patch


class ExperimentProtocolTests(unittest.TestCase):
    def test_config_cli_override_and_ablation(self):
        args = parse_args(['--config','configs/reference.json','--dataset','market1501',
                           '--output-dir','unused','--rounds','2','--ablation','no_temporal'])
        self.assertEqual(args.rounds,2)
        self.assertEqual(args.source_epochs,60)
        self.assertEqual(args.temporal_weight,0)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'invalid.json'
            config.write_text(json.dumps({'misspelled_setting':5}))
            with self.assertRaises(ValueError):
                parse_args(['--config',str(config)])

    def test_deferred_interventions_change_sampler_membership(self):
        mask = torch.tensor([False,True,True])
        self.assertTrue(torch.equal(deferred_batch_mask(mask),mask))
        self.assertFalse(deferred_batch_mask(mask,discard=True).any())
        for variant, weight, discarded in [('no_consistency',0.,False),('discard_deferred',0.1,True)]:
            args = parse_args(['--dataset','market1501','--output-dir','unused','--ablation',variant])
            self.assertEqual(args.deferred_weight,weight)
            self.assertEqual(args.discard_deferred,discarded)

    def test_size_strata_exclude_outliers(self):
        rows = cluster_size_diagnostics([0,0,1,1,1,1,-1],[2,2,3,3,4,4,9],
                                        [.8,.8,.5,.5,.5,.5,0],[1,1,1,1,3,3,0])
        self.assertEqual(rows[0]['samples'],2)
        self.assertEqual(rows[1]['pseudo_label_precision'],0.5)
        self.assertEqual(sum(r['samples'] for r in rows),6)

    def test_msmt_official_lists_and_explicit_trainval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'MSMT17_V1'
            for part,folder,pid in [('train','train',3),('val','train',4),('query','test',8),('gallery','test',8)]:
                image = root/folder/f'0000_00_01_{part}.jpg'
                image.parent.mkdir(parents=True,exist_ok=True)
                image.touch()
                (root/f'list_{part}.txt').write_text(f'{image.name} {pid}\n')
            train = MSMT17(directory,verbose=False)
            combined = MSMT17(directory,verbose=False,combine_trainval=True)
            self.assertEqual(len(train.train),1)
            self.assertEqual(len(combined.train),2)
            self.assertEqual(train.query[0][1],8)
            self.assertEqual(train.query[0][2],0)

    def test_manifest_matches_source_and_target_seed(self):
        args = SimpleNamespace(suite='main',seeds=[1,2,3],include_source=True,
            config='configs/reference.json',data_dir='data',source_root='logs/source',runs_dir='logs/test')
        runs = build_runs(args)
        self.assertEqual(len(runs),15)
        for run in runs[6:]:
            argv = run['argv']
            checkpoint = argv[argv.index('--checkpoint')+1]
            self.assertIn('seed_'+str(run['seed']),checkpoint)

    def test_zero_consistency_updates_bn_without_optimizer_step(self):
        from cgplr.training import adapt
        class BNEncoder(nn.Module):
            num_features = 4
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3,4,1)
                self.bn = nn.BatchNorm1d(4)
            def forward(self,x):
                return torch.nn.functional.normalize(self.bn(self.conv(x).mean((2,3))),dim=1)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(4):
                path = root/f'{i}.png'
                Image.fromarray(np.full((32,16,3),30+i*20,dtype=np.uint8)).save(path)
                rows.append((str(path),i//2,i%2))
            args = parse_args(['--dataset','market1501','--output-dir',str(root),
                '--ablation','no_consistency','--device','cpu','--rounds','1','--iters','1',
                '--batch-size','4','--instances','2','--workers','0','--height','32','--width','16',
                '--rho-high','1000000000'])
            model = BNEncoder()
            before = {key:value.detach().clone() for key,value in model.named_parameters()}
            with patch('cgplr.training.cluster_features',return_value=np.zeros(4,dtype=int)):
                adapt(model,SimpleNamespace(train=rows),args,root,torch.device('cpu'))
            result = json.loads((root/'rounds.json').read_text())[0]
            self.assertEqual(result['deferred'],4)
            self.assertEqual(result['optimizer_updates'],0)
            self.assertGreater(model.bn.num_batches_tracked.item(),0)
            for key,value in model.named_parameters():
                self.assertTrue(torch.equal(value,before[key]))


if __name__ == '__main__':
    unittest.main()
