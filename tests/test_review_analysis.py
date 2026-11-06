import unittest
from types import SimpleNamespace
import numpy as np
import json
from pathlib import Path
import tempfile
from cgplr.review_analysis import temporal_transition, deferred_cohort_transition
from examples.run_experiments import build_runs


def snapshot(labels, decisions, centers):
    labels = np.asarray(labels)
    return dict(filenames=np.array(['a','b','c','d']),raw_labels=labels,
                decisions=np.asarray(decisions),cluster_ids=np.unique(labels[labels>=0]),
                centroids=np.asarray(centers,dtype=np.float32))


class ReviewAnalysisTests(unittest.TestCase):
    def test_split_and_merge_separate_from_centroid_matching(self):
        old = snapshot([0,0,1,1],[3,3,1,1],[[1,0],[0,1]])
        new = snapshot([4,5,5,5],[1,1,1,1],[[1,0],[0,1]])
        row = temporal_transition(old,new,np.array([7,7,8,8]))
        self.assertEqual(row['split_clusters'],1)
        self.assertEqual(row['merged_clusters'],1)
        self.assertEqual(row['birth_clusters'],0)
        self.assertEqual(row['death_clusters'],0)
        self.assertEqual(row['split_samples'],2)

    def test_outlier_cohort_is_not_treated_as_newly_correct(self):
        old = snapshot([0,0,1,1],[3,3,1,1],[[1,0],[0,1]])
        new = snapshot([-1,-1,5,5],[0,0,1,1],[[0,1]])
        row = deferred_cohort_transition(old,new,np.array([7,7,8,8]))
        self.assertEqual(row['all']['n'],2)
        self.assertEqual(row['all']['next_outlier_fraction'],1.)
        self.assertIsNone(row['all']['next_raw_precision'])
        transition = temporal_transition(old,new,np.array([7,7,8,8]))
        self.assertEqual(transition['death_clusters'],1)
        new['filenames'] = new['filenames'][::-1]
        with self.assertRaises(ValueError):
            deferred_cohort_transition(old,new,np.array([7,7,8,8]))

    def test_all_suite_deduplicates_shared_controls(self):
        args = SimpleNamespace(suite='all',seeds=[1,2,3],include_source=True,
            config='configs/reference.json',data_dir='data',source_root='logs/source',runs_dir='logs/reviewer')
        runs = build_runs(args)
        keys = [(r['setting'],r['variant'],r['seed']) for r in runs]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(sum(r['variant']=='source' for r in runs),6)
        self.assertEqual(sum(r['variant']=='full' for r in runs),9)
        self.assertEqual(sum(r['variant']=='weak_view_consistency' for r in runs),9)

    def test_export_summarizes_complete_seeds_and_records_missing_outputs(self):
        from examples.run_review_experiments import export_results
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root/'reports'
            report.mkdir()
            runs = []
            for variant in ['full','no_consistency']:
                for seed in [1,2]:
                    folder = root/variant/str(seed)
                    folder.mkdir(parents=True)
                    config = dict(seed=seed,output_dir=str(folder),dataset='market1501',rounds=50,
                                  ablation=variant,deferred_weight=0.1 if variant=='full' else 0.)
                    (folder/'config.json').write_text(json.dumps(config))
                    (folder/'environment.json').write_text(json.dumps(dict(checkpoint_sha256=str(seed))))
                    (folder/'metrics.json').write_text(json.dumps(dict(R1=.8+seed*.01,mAP=.7+seed*.01)))
                    runs.append(dict(setting='MSMT_to_M',variant=variant,seed=seed,run_dir=str(folder)))
            runs.append(dict(setting='M_to_MSMT',variant='full',seed=1,run_dir=str(root/'missing')))
            export_results(runs,SimpleNamespace(report_dir=str(report),data_dir='data',seeds=[1,2],plot=False))
            coverage = json.loads((report/'coverage.json').read_text())
            self.assertEqual(coverage['completed_runs'],4)
            self.assertEqual(len(coverage['missing_runs']),1)
            result = json.loads((report/'MSMT_to_M'/'no_consistency.json').read_text())
            self.assertEqual(result['n'],2)
            self.assertIn('paired_comparison',result['mAP'])
            self.assertTrue((report/'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
