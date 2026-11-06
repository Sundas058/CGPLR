import numpy as np
from sklearn.metrics import roc_auc_score


def correctness(labels, identities, mapping=None):
    labels, identities = np.asarray(labels), np.asarray(identities)
    if labels.shape != identities.shape or labels.ndim != 1:
        raise ValueError('aligned one-dimensional assignments and identities required')
    if mapping is None:
        mapping = {}
        for cluster in np.unique(labels[labels >= 0]):
            ids, counts = np.unique(identities[labels == cluster], return_counts=True)
            mapping[int(cluster)] = int(ids[counts.argmax()])
    return np.array([label >= 0 and mapping.get(int(label)) == int(pid)
                     for label, pid in zip(labels, identities)]), mapping


def calibration(scores, correct, bins=10):
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    if scores.shape != correct.shape or scores.ndim != 1 or bins < 1:
        raise ValueError('aligned scores/correctness and positive bins required')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('scores must be finite in [0, 1]')
    rows, ece = [], 0.0
    for index in range(bins):
        low, high = index/bins, (index+1)/bins
        selected = (scores >= low) & ((scores <= high) if index == bins-1 else (scores < high))
        count = int(selected.sum())
        mean = float(scores[selected].mean()) if count else None
        precision = float(correct[selected].mean()) if count else None
        if count:
            ece += count/max(len(scores), 1) * abs(mean-precision)
        rows.append(dict(low=low, high=high, count=count, mean_score=mean, precision=precision))
    return dict(n=len(scores), ece=ece if len(scores) else None,
                brier=float(np.mean((scores-correct)**2)) if len(scores) else None,
                auc=float(roc_auc_score(correct, scores)) if len(np.unique(correct)) == 2 else None,
                bins=rows)


def decision_diagnostics(raw, refined, decisions, identities):
    original, mapping = correctness(raw, identities)
    corrected, _ = correctness(refined, identities, mapping)
    valid = np.asarray(raw) >= 0
    result = {}
    for decision, name in [(1, 'preserved'), (2, 'reassigned'), (3, 'deferred')]:
        mask = np.asarray(decisions) == decision
        result[name+'_count'] = int(mask.sum())
        result[name+'_rate'] = float(mask.sum()/valid.sum()) if valid.any() else None
        result[name+'_precision'] = float(corrected[mask].mean()) if mask.any() and decision != 3 else None
    reassigned_wrong = (np.asarray(decisions) == 2) & ~original
    result['correction_rate'] = float(corrected[reassigned_wrong].mean()) if reassigned_wrong.any() else None
    deferred = np.asarray(decisions) == 3
    result['deferred_original_error'] = float((~original[deferred]).mean()) if deferred.any() else None
    result['hard_coverage'] = float(np.isin(decisions, [1, 2]).sum()/valid.sum()) if valid.any() else None
    return result


def cluster_size_diagnostics(labels, identities, scores, decisions=None):

    labels, scores = np.asarray(labels), np.asarray(scores)
    correct, _ = correctness(labels, identities)
    clusters, sizes = np.unique(labels[labels >= 0], return_counts=True)
    rows = []
    for low, high in [(1,3),(4,7),(8,15),(16,31),(32,None)]:
        chosen = clusters[(sizes >= low) & ((sizes <= high) if high is not None else True)]
        mask = np.isin(labels,chosen)
        row = dict(minimum=low,maximum=high,clusters=int(len(chosen)),samples=int(mask.sum()),
                   pseudo_label_precision=float(correct[mask].mean()) if mask.any() else None,
                   mean_reliability=float(scores[mask].mean()) if mask.any() else None)
        if decisions is not None:
            row['deferred_fraction'] = float((np.asarray(decisions)[mask] == 3).mean()) if mask.any() else None
        rows.append(row)
    return rows


def retrieval_metrics(query_features, gallery_features, query, gallery, chunk_size=128, topk=10):

    import torch
    gallery_features = torch.as_tensor(gallery_features).float().cpu()
    query_features = torch.as_tensor(query_features).float().cpu()
    qids = np.array([row[1] for row in query])
    gids = np.array([row[1] for row in gallery])
    qcams = np.array([row[2] for row in query])
    gcams = np.array([row[2] for row in gallery])
    aps, cmcs, cases = [], [], []
    for start in range(0, len(query), chunk_size):
        distances = torch.cdist(query_features[start:start+chunk_size], gallery_features).square().numpy()
        for local, distance in enumerate(distances):
            i = start+local
            order = np.argsort(distance, kind='stable')
            order = order[~((gids[order] == qids[i]) & (gcams[order] == qcams[i])) & (gids[order] >= 0)]
            matches = gids[order] == qids[i]
            if not matches.any():
                continue
            ranks = np.flatnonzero(matches)
            aps.append(float(np.mean(np.arange(1, len(ranks)+1)/(ranks+1))))
            cmcs.append([bool(matches[:k].any()) for k in (1, 5, 10)])
            cases.append(dict(query=query[i][0], correct_top1=bool(matches[0]),
                              gallery=[gallery[j][0] for j in order[:topk]],
                              matches=matches[:topk].tolist()))
    if not aps:
        raise ValueError('no valid queries under the specified protocol')
    means = np.asarray(cmcs).mean(axis=0)
    return dict(mAP=float(np.mean(aps)), R1=float(means[0]), R5=float(means[1]), R10=float(means[2]),
                valid_queries=len(aps)), cases
