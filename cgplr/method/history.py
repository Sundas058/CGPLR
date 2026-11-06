from collections import deque
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class ClusterSnapshot:


    labels: torch.Tensor
    cluster_ids: torch.Tensor
    centroids: torch.Tensor


class ClusterHistory(object):


    def __init__(self, max_rounds=3):
        self.max_rounds = int(max_rounds)
        if self.max_rounds < 0:
            raise ValueError('max_rounds must be nonnegative')
        self._snapshots = deque(maxlen=max(self.max_rounds, 0))

    def __len__(self):
        return len(self._snapshots)

    def __iter__(self):
        return iter(self._snapshots)

    def clear(self):
        self._snapshots.clear()

    def append(self, labels, cluster_ids, centroids):
        if self.max_rounds <= 0:
            return
        snapshot = ClusterSnapshot(
            labels=torch.as_tensor(labels).detach().cpu().long().clone(),
            cluster_ids=torch.as_tensor(cluster_ids).detach().cpu().long().clone(),
            centroids=torch.as_tensor(centroids).detach().cpu().float().clone(),
        )
        self._snapshots.append(snapshot)


def greedy_centroid_alignment(previous_cluster_ids, previous_centroids, current_cluster_ids, current_centroids):

    previous_cluster_ids = torch.as_tensor(previous_cluster_ids).long()
    current_cluster_ids = torch.as_tensor(current_cluster_ids).long()
    previous_centroids = torch.as_tensor(previous_centroids).float()
    current_centroids = torch.as_tensor(current_centroids).float()

    if previous_centroids.numel() == 0 or current_centroids.numel() == 0:
        return {}

    distances = torch.cdist(previous_centroids, current_centroids, p=2) ** 2
    pairs = []
    for previous_index in range(distances.size(0)):
        for current_index in range(distances.size(1)):
            pairs.append((
                float(distances[previous_index, current_index].item()),
                previous_index,
                current_index,
            ))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    used_previous = set()
    used_current = set()
    mapping = {}
    for _, previous_index, current_index in pairs:
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        mapping[int(previous_cluster_ids[previous_index].item())] = int(
            current_cluster_ids[current_index].item()
        )
    return mapping


def minimum_cost_centroid_alignment(previous_cluster_ids, previous_centroids,
                                    current_cluster_ids, current_centroids):

    if len(previous_cluster_ids) == 0 or len(current_cluster_ids) == 0:
        return {}
    distances = torch.cdist(torch.as_tensor(previous_centroids).float().cpu(),
                            torch.as_tensor(current_centroids).float().cpu()).square()
    rows, columns = linear_sum_assignment(distances.numpy())
    return {int(previous_cluster_ids[r]): int(current_cluster_ids[c])
            for r, c in zip(rows, columns)}


def temporal_assignment_stability(labels, cluster_ids, centroids, history,
                                  outlier_label=-1, mode='centroid'):

    labels = torch.as_tensor(labels).long().cpu()
    stability = torch.zeros(labels.size(0), dtype=torch.float32)
    if mode not in ('centroid', 'overlap'):
        raise ValueError('temporal mode must be centroid or overlap')
    if history is None or len(history) == 0:
        return stability

    compared_rounds = 0
    observations = torch.zeros_like(stability)
    for snapshot in history:
        if len(snapshot.labels) != len(labels):
            raise ValueError('history requires identical ordered sample sets across rounds')
        if mode == 'overlap':

            
            valid = (labels != outlier_label) & (snapshot.labels != outlier_label)
            for cluster_id in torch.unique(labels[valid]):
                members = labels == cluster_id
                for old_id in torch.unique(snapshot.labels[members & valid]):
                    old_members = snapshot.labels == old_id
                    shared = (members & old_members).sum().float()
                    union = (members | old_members).sum().float()
                    mask = members & old_members & valid
                    stability[mask] += shared / union
            observations[valid] += 1
            continue
        mapping = minimum_cost_centroid_alignment(
            snapshot.cluster_ids,
            snapshot.centroids,
            cluster_ids,
            centroids,
        )
        compared_rounds += 1
        previous_labels = snapshot.labels
        max_len = min(labels.size(0), previous_labels.size(0))
        for index in range(max_len):
            previous_label = int(previous_labels[index].item())
            current_label = int(labels[index].item())
            if previous_label == outlier_label or current_label == outlier_label:
                continue
            if mapping.get(previous_label) == current_label:
                stability[index] += 1.0

    if mode == 'overlap':
        return stability / observations.clamp_min(1)
    if compared_rounds == 0:
        return stability
    return stability / float(compared_rounds)
