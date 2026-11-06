import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.run_experiments import build_runs


def run_directory(run):
    if 'run_dir' in run:
        return Path(run['run_dir'])
    argv = run['argv']
    return Path(argv[argv.index('--output-dir')+1])


def export_results(runs, args):
    from examples.analyze_revision import diagnose, aggregate
    report = Path(args.report_dir)
    groups, missing, rows, round_rows, size_rows = {}, [], [], [], []
    for run in runs:
        if run['variant'] == 'source':
            continue
        folder = run_directory(run)
        if not (folder/'metrics.json').exists():
            missing.append(dict(setting=run['setting'],variant=run['variant'],seed=run['seed'],run=str(folder)))
            continue
        groups.setdefault((run['setting'],run['variant']),[]).append(str(folder))
        metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8-sig'))
        timing_file = folder/'timing.json'
        timing = json.loads(timing_file.read_text(encoding='utf-8-sig')) if timing_file.exists() else {}
        row = dict(setting=run['setting'],variant=run['variant'],seed=run['seed'],run=str(folder))
        for key in ['R1','mAP','forward_batch_ms','encoder_fps','encoder_parameters',
                    'query_extraction_seconds','gallery_extraction_seconds','gallery_ranking_seconds']:
            row[key] = metrics.get(key,'N/A')
        for key in ['source_wall_seconds_from_checkpoint','adaptation_wall_seconds','source_plus_adaptation_seconds',
                    'feature_extraction_seconds','clustering_seconds','reliability_seconds','refinement_seconds',
                    'optimization_seconds','peak_training_cuda_allocated_bytes','sampled_process_rss_peak_bytes']:
            row[key] = timing.get(key,'N/A')
        rows.append(row)
        if list(folder.glob('round_*.npz')):
            diagnose(SimpleNamespace(run=str(folder),data_dir=args.data_dir,
                output=str(folder/'diagnostics'),bins=10,last_rounds=5,plot=args.plot))
            diagnostics = json.loads((folder/'diagnostics'/'diagnostics.json').read_text())
            for diagnostic in diagnostics:
                base = dict(setting=run['setting'],variant=run['variant'],seed=run['seed'],round=diagnostic['round'])
                record = dict(base)
                for key in ['n','ece','auc','brier']:
                    record[key] = diagnostic['calibration'].get(key)
                record.update(diagnostic['decisions'])
                record.update({'temporal_'+k:v for k,v in diagnostic['temporal_summary'].items()})
                for key in ['split_clusters','merged_clusters','birth_clusters','death_clusters',
                            'valid_pair_samples','matched_samples','correct_both_but_unmatched_fraction']:
                    record[key] = diagnostic.get('temporal_transition',{}).get(key)
                for cohort in ['all','originally_wrong','originally_correct']:
                    values = diagnostic.get('deferred_cohort_transition',{}).get(cohort,{})
                    for key in ['n','next_outlier_fraction','next_raw_precision','next_hard_selected_fraction']:
                        record['deferred_'+cohort+'_'+key] = values.get(key)
                round_rows.append(record)
                size_rows.extend(dict(base,**stratum) for stratum in diagnostic['size_strata'])
    summaries = []
    for (setting,variant),paths in groups.items():
        output = report/setting/(variant+'.json')
        output.parent.mkdir(parents=True,exist_ok=True)
        baseline = groups.get((setting,'full')) if variant != 'full' else None
        paired = baseline if baseline and len(paths)==len(args.seeds)==len(baseline) else None
        params = SimpleNamespace(runs=paths,baselines=paired,output=str(output))
        reason = None
        try:
            aggregate(params)
        except ValueError as error:
            if not paired:
                raise
            reason = str(error)
            params.baselines = None
            aggregate(params)
        result = json.loads(output.read_text())
        row = dict(setting=setting,variant=variant,n=result['n'],expected_seeds=len(args.seeds))
        for metric in ['R1','mAP']:
            row[metric+'_mean'] = result[metric]['mean']
            row[metric+'_sd'] = result[metric]['sample_std']
            row[metric+'_paired_p'] = result[metric].get('paired_comparison',{}).get('two_sided_paired_t_p')
        row['paired_comparison_note'] = reason or ('N/A' if not paired else 'Matched protocol; paired against full')
        summaries.append(row)
    for name,table in [('per_run.csv',rows),('summary.csv',summaries),
                       ('round_diagnostics.csv',round_rows),('cluster_size_strata.csv',size_rows)]:
        if table:
            with (report/name).open('w',newline='',encoding='utf-8') as file:
                writer = csv.DictWriter(file,fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows({k:'N/A' if v is None else v for k,v in row.items()} for row in table)
    (report/'coverage.json').write_text(json.dumps(dict(completed_runs=len(rows),missing_runs=missing,
        units='Retrieval means/SD are fractions; multiply by 100 for percentage points'),indent=2),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['main','stagewise','mechanism','sensitivity','safety','all'],default='all')
    p.add_argument('--config',default='configs/reference.json')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--source-root',default='logs/source')
    p.add_argument('--runs-dir',default='logs/reviewer')
    p.add_argument('--report-dir',default='reports/reviewer')
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--include-source',action='store_true')
    p.add_argument('--competitor-manifest',help='JSON list of setting/variant/seed/argv/run_dir records')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--plot',action='store_true')
    args = p.parse_args()
    if args.execute and args.analyze_only:
        p.error('--execute and --analyze-only are mutually exclusive')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be nonempty and distinct')
    runs = build_runs(args)
    if args.competitor_manifest:
        external = json.loads(Path(args.competitor_manifest).read_text(encoding='utf-8-sig'))
        if not isinstance(external,list):
            raise ValueError('Competitor manifest must be a list')
        for entry in external:
            if not {'setting','variant','seed','argv','run_dir'} <= set(entry):
                raise ValueError('Incomplete competitor manifest entry')
            if not isinstance(entry['argv'],list) or not all(isinstance(v,str) for v in entry['argv']):
                raise ValueError('Competitor argv must be a list of strings')
        runs.extend(external)
    keys = [(r['setting'],r['variant'],r['seed']) for r in runs]
    paths = [str(run_directory(r).resolve()).casefold() for r in runs]
    if len(set(keys)) != len(keys) or len(set(paths)) != len(paths):
        raise ValueError('Duplicate experiment identity or output directory')
    report = Path(args.report_dir)
    report.mkdir(parents=True,exist_ok=True)
    (report/'manifest.json').write_text(json.dumps(dict(count=len(runs),runs=runs),indent=2),encoding='utf-8')
    print(f'{len(runs)} unique commands: {report / "manifest.json"}',flush=True)
    if args.execute:
        for run in runs:
            subprocess.run(run['argv'],cwd=ROOT,check=True)
    if args.execute or args.analyze_only:
        export_results(runs,args)


if __name__ == '__main__':
    main()
