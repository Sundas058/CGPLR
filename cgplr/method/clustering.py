"""Target-domain clustering utilities for CGPLR.

This module implements the feature normalization, DBSCAN pseudo-label
generation, valid-cluster filtering, and centroid computation described in
Section 3.4 of the manuscript.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def valid_cluster_mask(pseudo_labels, outlier_label=-1):
    """Return a boolean mask selecting samples assigned to valid clusters."""
    labels = torch.as_tensor(pseudo_labels)
    return labels != outlier_label


def compute_cluster_centroids(features, pseudo_labels, outlier_label=-1):
    """Compute L2-normalized centroids for valid target clusters.

    Args:
        features: Target embeddings with shape ``[num_samples, feature_dim]``.
        pseudo_labels: Cluster labels produced by DBSCAN.
        outlier_label: Label used by DBSCAN for outliers.

    Returns:
        A tuple ``(cluster_ids, centroids)``. ``cluster_ids`` preserves the
        original cluster label values, and ``centroids`` has shape
        ``[num_clusters, feature_dim]``.
    """
    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels).long()
    cluster_ids = torch.unique(labels[labels != outlier_label], sorted=True)

    if cluster_ids.numel() == 0:
        empty = features.new_empty((0, features.size(1)))
        return cluster_ids, empty

    centroids = []
    for cluster_id in cluster_ids:
        members = features[labels == cluster_id]
        centroids.append(members.mean(dim=0))

    centroids = F.normalize(torch.stack(centroids, dim=0), dim=1)
    return cluster_ids, centroids


def compact_cluster_labels(pseudo_labels, outlier_label=-1):
    """Map valid cluster labels to contiguous ids while preserving outliers."""
    labels = torch.as_tensor(pseudo_labels).long().clone()
    compact = labels.new_full(labels.size(), outlier_label)
    valid_ids = torch.unique(labels[labels != outlier_label], sorted=True)
    mapping = {}
    for new_id, old_id in enumerate(valid_ids.tolist()):
        compact[labels == int(old_id)] = int(new_id)
        mapping[int(old_id)] = int(new_id)
    return compact, mapping


def cluster_with_dbscan(distance_matrix, eps=0.55, min_samples=4, n_jobs=-1):
    """Generate target-domain pseudo labels with DBSCAN.

    ``distance_matrix`` is expected to be a precomputed distance matrix, such
    as the Jaccard distance used by the existing training scripts.
    """
    distances = np.asarray(distance_matrix)
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed',
        n_jobs=n_jobs,
    )
    return clusterer.fit_predict(distances)
