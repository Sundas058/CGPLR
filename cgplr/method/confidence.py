"""Sample-level confidence estimation for CGPLR.

The final confidence score follows Eq. (13): a weighted combination of
centroid-based confidence and local neighborhood consistency.
"""

import torch
import torch.nn.functional as F

from .clustering import compute_cluster_centroids, valid_cluster_mask


def _centroid_lookup(cluster_ids):
    return {int(cluster_id.item()): index for index, cluster_id in enumerate(cluster_ids)}


def _centroid_confidence(features, labels, cluster_ids, centroids, temperature):
    lookup = _centroid_lookup(cluster_ids)
    confidence = features.new_zeros(labels.size(0))

    for sample_index, label in enumerate(labels.tolist()):
        centroid_index = lookup.get(int(label))
        if centroid_index is None:
            continue
        distance = torch.sum((features[sample_index] - centroids[centroid_index]) ** 2)
        confidence[sample_index] = torch.exp(-distance / temperature)

    return confidence.clamp_(0.0, 1.0)


def _neighborhood_consistency(features, labels, neighbors):
    similarities = torch.mm(features, features.t())
    similarities.fill_diagonal_(-1.0)
    k = min(int(neighbors), max(features.size(0) - 1, 1))
    neighbor_indices = similarities.topk(k=k, dim=1).indices
    neighbor_labels = labels[neighbor_indices]
    return neighbor_labels.eq(labels.view(-1, 1)).float().mean(dim=1)


def estimate_sample_confidence(
    features,
    pseudo_labels,
    alpha=0.5,
    temperature=1.0,
    neighbors=5,
    outlier_label=-1,
):
    """Estimate confidence weights for pseudo-labeled target samples.

    Args:
        features: Target embeddings with shape ``[num_samples, feature_dim]``.
        pseudo_labels: Initial cluster assignments.
        alpha: Balance between centroid confidence and neighbor consistency.
        temperature: Temperature used by Eq. (11).
        neighbors: Local neighborhood size ``K`` in Eq. (12).
        outlier_label: Label used for unclustered DBSCAN outliers.

    Returns:
        Confidence weights in ``[0, 1]`` for every sample. Outliers receive 0.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')

    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels, device=features.device).long()
    valid_mask = valid_cluster_mask(labels, outlier_label=outlier_label).to(features.device)
    confidence = features.new_zeros(labels.size(0))

    if valid_mask.sum() <= 1:
        return confidence

    valid_features = features[valid_mask]
    valid_labels = labels[valid_mask]
    cluster_ids, centroids = compute_cluster_centroids(
        valid_features,
        valid_labels,
        outlier_label=outlier_label,
    )

    centroid_scores = _centroid_confidence(
        valid_features,
        valid_labels,
        cluster_ids.to(features.device),
        centroids.to(features.device),
        temperature,
    )
    neighbor_scores = _neighborhood_consistency(valid_features, valid_labels, neighbors)
    confidence[valid_mask] = alpha * centroid_scores + (1.0 - alpha) * neighbor_scores
    return confidence.clamp_(0.0, 1.0)
