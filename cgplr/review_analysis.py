import numpy as np
from .diagnostics import correctness
from .method.history import minimum_cost_centroid_alignment


def temporal_transition(previous, current, identities):

    if not np.array_equal(previous['filenames'],current['filenames']):
        raise ValueError('Temporal diagnostics require identical ordered filenames')
    old, new = previous['raw_labels'], current['raw_labels']
    valid_pair = (old >= 0) & (new >= 0)
    pairs = np.unique(np.stack([old[valid_pair],new[valid_pair]],axis=1),axis=0)
    old_ids, new_ids = np.unique(old[old >= 0]), np.unique(new[new >= 0])
    outgoing = {int(k):set() for k in old_ids}
    incoming = {int(k):set() for k in new_ids}
    for before, after in pairs:
        outgoing[int(before)].add(int(after))
        incoming[int(after)].add(int(before))
    split = {k for k,v in outgoing.items() if len(v)>1}
    merge = {k for k,v in incoming.items() if len(v)>1}
    mapping = minimum_cost_centroid_alignment(previous['cluster_ids'],previous['centroids'],
        current['cluster_ids'],current['centroids'])
    matched = np.array([a >= 0 and b >= 0 and mapping.get(int(a)) == int(b) for a,b in zip(old,new)])
    old_correct, _ = correctness(old,identities)
    new_correct, _ = correctness(new,identities)
    stable_correct = valid_pair & old_correct & new_correct
    output = dict(previous_clusters=len(old_ids),current_clusters=len(new_ids),
        split_clusters=len(split),merged_clusters=len(merge),
        birth_clusters=sum(not v for v in incoming.values()),
        death_clusters=sum(not v for v in outgoing.values()),
        current_outliers=int((new < 0).sum()), historical_outliers=int((old < 0).sum()),
        valid_pair_samples=int(valid_pair.sum()),matched_samples=int(matched.sum()),
        correct_both_rounds_samples=int(stable_correct.sum()),
        correct_both_but_unmatched_fraction=float((~matched[stable_correct]).mean()) if stable_correct.any() else None)
    for name, mask in [('split',valid_pair & np.isin(old,list(split))),
                       ('merge',valid_pair & np.isin(new,list(merge))),
                       ('unmatched',valid_pair & ~matched)]:
        output[name+'_samples'] = int(mask.sum())
        output[name+'_precision'] = float(new_correct[mask].mean()) if mask.any() else None
    return output


def deferred_cohort_transition(previous, current, identities):

    if not np.array_equal(previous['filenames'],current['filenames']):
        raise ValueError('Deferred diagnostics require identical ordered filenames')
    old_correct, _ = correctness(previous['raw_labels'],identities)
    new_correct, _ = correctness(current['raw_labels'],identities)
    deferred = previous['decisions'] == 3
    valid = current['raw_labels'] >= 0
    rows = {}
    for name,mask in [('all',deferred),('originally_wrong',deferred & ~old_correct),
                      ('originally_correct',deferred & old_correct)]:
        count = int(mask.sum())
        observed = mask & valid
        rows[name] = dict(n=count,next_clustered=int(observed.sum()),
            next_outlier_fraction=float((~valid[mask]).mean()) if count else None,
            next_raw_precision=float(new_correct[observed].mean()) if observed.any() else None,
            next_hard_selected_fraction=float(np.isin(current['decisions'][mask],[1,2]).mean()) if count else None)
    return rows
