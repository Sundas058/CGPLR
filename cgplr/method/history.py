"""Temporal state for CGPLR adaptation rounds.

The updated manuscript estimates confidence from both current clustering
structure and assignment stability across recent adaptation rounds. This
module keeps the small amount of history needed for that temporal term.
"""

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class ClusterSnapshot:
    """Pseudo labels and centroids from one adaptation round."""

    labels: torch.Tensor
    cluster_ids: torch.Tensor
    centroids: torch.Tensor


class ClusterHistory(object):
    """Fixed-window history of clustering snapshots.

    Args:
        max_rounds: Number of previous rounds retained for temporal stability.
    """

    def __init__(self, max_rounds=3):
        self.max_rounds = int(max_rounds)
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
    """Align historical cluster ids to current cluster ids by nearest centroids.

    The paper formulates a maximum-cardinality one-to-one alignment. For code
    portability and to avoid requiring SciPy at runtime, this implementation
    uses a deterministic greedy minimum-distance matching. It is sufficient for
    the temporal stability signal and keeps the dependency surface small.
    """
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


def temporal_assignment_stability(labels, cluster_ids, centroids, history, outlier_label=-1):
    """Compute assignment stability over the retained cluster history."""
    labels = torch.as_tensor(labels).long().cpu()
    stability = torch.zeros(labels.size(0), dtype=torch.float32)
    if history is None or len(history) == 0:
        return stability

    compared_rounds = 0
    for snapshot in history:
        mapping = greedy_centroid_alignment(
            snapshot.cluster_ids,
            snapshot.centroids,
            cluster_ids,
            centroids,
        )
        if not mapping:
            continue
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

    if compared_rounds == 0:
        return stability
    return stability / float(compared_rounds)
