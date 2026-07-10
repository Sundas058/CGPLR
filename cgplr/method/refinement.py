"""Confidence-guided pseudo-label refinement for CGPLR."""

import torch
import torch.nn.functional as F

from .clustering import compute_cluster_centroids, valid_cluster_mask


def partition_by_confidence(confidence, threshold=0.6):
    """Return high-confidence and uncertain masks following Eq. (14)."""
    confidence = torch.as_tensor(confidence).float()
    high_confidence = confidence >= threshold
    uncertain = confidence < threshold
    return high_confidence, uncertain


def refine_pseudo_labels(features, pseudo_labels, confidence, threshold=0.6, outlier_label=-1):
    """Preserve high-confidence labels and reassign uncertain samples.

    Uncertain valid samples are assigned to the nearest cluster centroid,
    matching Eq. (16)-Eq. (17) in the manuscript. DBSCAN outliers remain
    unchanged by default.
    """
    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels, device=features.device).long()
    confidence = torch.as_tensor(confidence, device=features.device).float()

    valid_mask = valid_cluster_mask(labels, outlier_label=outlier_label).to(features.device)
    high_confidence, uncertain = partition_by_confidence(confidence, threshold)
    uncertain_valid = valid_mask & uncertain.to(features.device)

    refined = labels.clone()
    if uncertain_valid.sum() == 0:
        return refined

    cluster_ids, centroids = compute_cluster_centroids(
        features[valid_mask],
        labels[valid_mask],
        outlier_label=outlier_label,
    )
    centroids = centroids.to(features.device)
    cluster_ids = cluster_ids.to(features.device)

    distances = torch.cdist(features[uncertain_valid], centroids, p=2) ** 2
    nearest = distances.argmin(dim=1)
    refined[uncertain_valid] = cluster_ids[nearest]
    refined[valid_mask & high_confidence.to(features.device)] = labels[
        valid_mask & high_confidence.to(features.device)
    ]
    return refined
