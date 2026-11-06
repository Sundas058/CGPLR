import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys


def build_runs(args):
    variants = {
        'main': {'full': []},
        'stagewise': {name: ['--ablation',name] for name in
                      ['clustering','centroid','neighborhood','refinement','full']},
        'mechanism': {name: ['--ablation',name] for name in
                      ['full','no_margin','no_temporal','global_thresholds','forced_reassignment',
                       'discard_deferred','no_consistency','equal_weights','overlap_history','candidate_guard']},
        'safety': {
            'full': [],
            'no_consistency': ['--ablation','no_consistency'],
            'discard_deferred': ['--ablation','discard_deferred'],
            'weak_view_consistency': ['--strong-color-jitter','0','--strong-grayscale','0','--strong-erasing','0'],
            'consistency_0.05': ['--deferred-weight','0.05'],
            'consistency_0.2': ['--deferred-weight','0.2'],
            'consistency_0.4': ['--deferred-weight','0.4'],
        },
    }
    if args.suite == 'all':
        selected = {}
        for suite in ['main','stagewise','mechanism','safety','sensitivity']:
            child = argparse.Namespace(**vars(args))
            child.suite, child.include_source = suite, False
            for run in build_runs(child):
                key = (run['setting'],run['variant'],run['seed'])
                selected.setdefault(key,run)
        sources = []
        if args.include_source:
            child = argparse.Namespace(**vars(args))
            child.suite = 'main'
            sources = [run for run in build_runs(child) if run['variant'] == 'source']
        return sources + list(selected.values())
    if args.suite == 'sensitivity':
        selected = {}
        grid = {'alpha':[0.25,0.5,0.75], 'beta':[0.25,0.5,0.75], 'neighbors':[3,5,10],
                'centroid-temperature':[0.25,0.5,1.0], 'margin-temperature':[0.05,0.1,0.2],
                'reassignment-temperature':[0.25,0.5,1.0], 'temporal-window':[0,1,2,3,4],
                'rho-high':[0.8,1.,1.2], 'rho-low':[0.8,1.,1.2],
                'evidence-gain':[0.05,0.1,0.2], 'neighbor-support':[0.4,0.6,0.8],
                'reassigned-weight':[0.5,0.7,1.0], 'dispersion-eps':[1e-7,1e-6,1e-5],
                'deferred-weight':[0,0.05,0.1,0.2], 'triplet-weight':[0.5,1.,2.],
                'strong-erasing':[0,0.25,0.5], 'strong-erasing-max':[0.1,0.2,0.4],
                'strong-color-jitter':[0,0.2,0.4]}
        for name, values in grid.items():
            for value in values:
                selected[name+'_'+str(value)] = ['--'+name,str(value)]
        for temporal in [0,0.1,0.2,0.3]:
            selected['fusion_temporal_'+str(temporal)] = ['--structural-weight',str((1-temporal)*5/8),
                '--margin-weight',str((1-temporal)*3/8),'--temporal-weight',str(temporal)]
        for ratio in [0.5,0.625,0.75]:
            selected['fusion_structural_'+str(ratio)] = ['--structural-weight',str(0.8*ratio),
                '--margin-weight',str(0.8*(1-ratio)),'--temporal-weight','0.2']
    else:
        selected = variants[args.suite]
    transfers = [('M_to_MSMT','msmt17','market1501'),('MSMT_to_M','market1501','msmt17'),
                 ('M_to_CUHK','cuhk03-np-d','market1501')]
    runs = []
    if args.include_source:
        for dataset, seed in itertools.product(['market1501','msmt17'],args.seeds):
            argv = [sys.executable,'examples/train_cgplr.py','--config',args.config,'--stage','source',
                    '--dataset',dataset,'--imagenet','--data-dir',args.data_dir,'--seed',str(seed),
                    '--output-dir',str(Path(args.source_root)/dataset/f'seed_{seed}')]
            runs.append(dict(setting=dataset,variant='source',seed=seed,argv=argv))
    for (setting,target,source),(variant,flags),seed in itertools.product(transfers,selected.items(),args.seeds):
        checkpoint = Path(args.source_root)/source/f'seed_{seed}'/'source.pth'
        argv = [sys.executable,'examples/train_cgplr.py','--config',args.config,'--stage','adapt',
                '--dataset',target,'--data-dir',args.data_dir,'--checkpoint',str(checkpoint),
                '--seed',str(seed),'--output-dir',str(Path(args.runs_dir)/setting/variant/f'seed_{seed}')]+flags
        runs.append(dict(setting=setting,variant=variant,seed=seed,argv=argv))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['main','stagewise','mechanism','sensitivity','safety','all'],default='main')
    p.add_argument('--config',default='configs/reference.json')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source-root',default='logs/source')
    p.add_argument('--runs-dir',default='logs/experiments')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--output',default='experiment_manifest.json')
    p.add_argument('--include-source',action='store_true')
    p.add_argument('--execute',action='store_true')
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct')
    runs = build_runs(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(suite=args.suite,count=len(runs),runs=runs),indent=2),encoding='utf-8')
    print(f'{len(runs)} commands written to {output}')
    if args.execute:
        for run in runs:
            subprocess.run(run['argv'],check=True)


if __name__ == '__main__':
    main()
