"""Adaptive selective pseudo-label refinement for CGPLR."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .clustering import compute_cluster_centroids, valid_cluster_mask
from .confidence import ConfidenceResult


PRESERVE = 1
REASSIGN = 2
DEFER = 3


@dataclass
class RefinementResult:
    """Output of cluster-adaptive selective refinement."""

    refined_labels: torch.Tensor
    decisions: torch.Tensor
    learning_weights: torch.Tensor
    preserve_mask: torch.Tensor
    reassign_mask: torch.Tensor
    defer_mask: torch.Tensor
    upper_thresholds: torch.Tensor
    lower_thresholds: torch.Tensor
    assignment_evidence: torch.Tensor
    best_alternatives: torch.Tensor
    evidence_gain: torch.Tensor
    alternative_support: torch.Tensor


def partition_by_confidence(confidence, threshold=0.6):
    """Return high-confidence and uncertain masks following Eq. (14)."""
    confidence = torch.as_tensor(confidence).float()
    high_confidence = confidence >= threshold
    uncertain = confidence < threshold
    return high_confidence, uncertain


def cluster_adaptive_thresholds(labels, confidence, cluster_ids=None, rho_high=1.0, rho_low=1.0, eps=1e-6):
    """Compute median/MAD upper and lower thresholds for each cluster."""
    labels = torch.as_tensor(labels).long()
    confidence = torch.as_tensor(confidence).float()
    if cluster_ids is None:
        cluster_ids = torch.unique(labels[labels >= 0], sorted=True)
    else:
        cluster_ids = torch.as_tensor(cluster_ids).long()

    thresholds_high = confidence.new_zeros(labels.size(0))
    thresholds_low = confidence.new_zeros(labels.size(0))
    for cluster_id in cluster_ids.tolist():
        mask = labels == int(cluster_id)
        if mask.sum() == 0:
            continue
        values = confidence[mask]
        median = values.median()
        mad = (values - median).abs().median()
        dispersion = torch.clamp(mad, min=eps)
        thresholds_high[mask] = torch.clamp(median + rho_high * dispersion, 0.0, 1.0)
        thresholds_low[mask] = torch.clamp(median - rho_low * dispersion, 0.0, 1.0)
    return thresholds_high, thresholds_low


def assignment_evidence(features, labels, cluster_ids, centroids, neighbors=5, beta=0.5, temperature=0.5):
    """Compute fused geometric and neighborhood support for every cluster."""
    if temperature <= 0:
        raise ValueError('reassignment temperature must be positive')
    if not 0.0 <= beta <= 1.0:
        raise ValueError('beta must be in [0, 1]')

    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(labels, device=features.device).long()
    cluster_ids = torch.as_tensor(cluster_ids, device=features.device).long()
    centroids = F.normalize(torch.as_tensor(centroids, device=features.device).float(), dim=1)

    if cluster_ids.numel() == 0:
        return features.new_empty((features.size(0), 0))

    distances = torch.cdist(features, centroids, p=2) ** 2
    geometric = torch.exp(-distances / temperature).clamp_(0.0, 1.0)

    if features.size(0) <= 1:
        support = torch.zeros_like(geometric)
    else:
        similarities = torch.mm(features, features.t())
        similarities.fill_diagonal_(-1.0)
        k = min(int(neighbors), features.size(0) - 1)
        neighbor_indices = similarities.topk(k=k, dim=1).indices
        neighbor_labels = labels[neighbor_indices]
        support = torch.zeros_like(geometric)
        for column, cluster_id in enumerate(cluster_ids.tolist()):
            support[:, column] = neighbor_labels.eq(int(cluster_id)).float().mean(dim=1)

    return beta * geometric + (1.0 - beta) * support


def adaptive_selective_refinement(
    features,
    pseudo_labels,
    confidence_result,
    rho_high=1.0,
    rho_low=1.0,
    dispersion_eps=1e-6,
    reassignment_beta=0.5,
    reassignment_temperature=0.5,
    evidence_gain=0.1,
    neighbor_support=0.6,
    reassigned_weight=0.7,
    neighbors=5,
    outlier_label=-1,
    eps=1e-12,
):
    """Preserve, reassign or defer valid samples following Eq. (10)-Eq. (17)."""
    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels, device=features.device).long()
    if isinstance(confidence_result, ConfidenceResult):
        confidence = confidence_result.weights.to(features.device)
        cluster_ids = confidence_result.cluster_ids.to(features.device)
        centroids = confidence_result.centroids.to(features.device)
    else:
        confidence = torch.as_tensor(confidence_result, device=features.device).float()
        cluster_ids, centroids = compute_cluster_centroids(features, labels, outlier_label=outlier_label)
        cluster_ids = cluster_ids.to(features.device)
        centroids = centroids.to(features.device)

    valid_mask = valid_cluster_mask(labels, outlier_label=outlier_label).to(features.device)
    refined = labels.clone()
    refined[~valid_mask] = outlier_label
    decisions = labels.new_full(labels.size(), DEFER)
    decisions[~valid_mask] = 0
    learning_weights = confidence.new_zeros(labels.size(0))
    best_alternatives = labels.new_full(labels.size(), outlier_label)
    gain_scores = confidence.new_zeros(labels.size(0))
    support_scores = confidence.new_zeros(labels.size(0))

    high_thresholds, low_thresholds = cluster_adaptive_thresholds(
        labels,
        confidence,
        cluster_ids=cluster_ids,
        rho_high=rho_high,
        rho_low=rho_low,
        eps=dispersion_eps,
    )
    high_thresholds = high_thresholds.to(features.device)
    low_thresholds = low_thresholds.to(features.device)

    preserve_mask = valid_mask & (confidence >= high_thresholds)
    decisions[preserve_mask] = PRESERVE
    learning_weights[preserve_mask] = confidence[preserve_mask]

    valid_indices = torch.where(valid_mask)[0]
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]
    evidence = assignment_evidence(
        valid_features,
        valid_labels,
        cluster_ids,
        centroids,
        neighbors=neighbors,
        beta=reassignment_beta,
        temperature=reassignment_temperature,
    )

    if evidence.numel() > 0 and cluster_ids.numel() > 1:
        label_to_column = {int(cluster_id.item()): column for column, cluster_id in enumerate(cluster_ids)}
        current_columns = torch.tensor(
            [label_to_column.get(int(label.item()), -1) for label in valid_labels],
            dtype=torch.long,
            device=features.device,
        )
        valid_current = current_columns >= 0
        current_evidence = evidence.new_zeros(valid_labels.size(0))
        current_evidence[valid_current] = evidence[
            torch.arange(valid_labels.size(0), device=features.device)[valid_current],
            current_columns[valid_current],
        ]
        alternative_evidence = evidence.clone()
        alternative_evidence[
            torch.arange(valid_labels.size(0), device=features.device)[valid_current],
            current_columns[valid_current],
        ] = -1.0
        best_columns = alternative_evidence.argmax(dim=1)
        best_evidence = evidence[torch.arange(valid_labels.size(0), device=features.device), best_columns]
        valid_gain = best_evidence - current_evidence

        best_alternatives[valid_indices] = cluster_ids[best_columns]
        gain_scores[valid_indices] = valid_gain
        support_scores[valid_indices] = best_evidence

        intermediate = (
            valid_mask
            & (confidence >= low_thresholds)
            & (confidence < high_thresholds)
        )
        reassign_mask = (
            intermediate
            & (gain_scores >= evidence_gain)
            & (support_scores >= neighbor_support)
        )
        refined[reassign_mask] = best_alternatives[reassign_mask]
        decisions[reassign_mask] = REASSIGN
        original_support = current_evidence.new_zeros(labels.size(0))
        original_support[valid_indices] = current_evidence
        correction_confidence = support_scores / (support_scores + original_support + eps)
        learning_weights[reassign_mask] = reassigned_weight * correction_confidence[reassign_mask]
    else:
        reassign_mask = torch.zeros_like(valid_mask)

    defer_mask = valid_mask & ~(preserve_mask | reassign_mask)
    decisions[defer_mask] = DEFER
    refined[defer_mask] = outlier_label

    return RefinementResult(
        refined_labels=refined,
        decisions=decisions,
        learning_weights=learning_weights.clamp_(0.0, 1.0),
        preserve_mask=preserve_mask,
        reassign_mask=reassign_mask,
        defer_mask=defer_mask,
        upper_thresholds=high_thresholds,
        lower_thresholds=low_thresholds,
        assignment_evidence=evidence,
        best_alternatives=best_alternatives,
        evidence_gain=gain_scores,
        alternative_support=support_scores,
    )


def refine_pseudo_labels(features, pseudo_labels, confidence, threshold=0.6, outlier_label=-1):
    """Preserve high-confidence labels and reassign uncertain samples.

    Uncertain valid samples are assigned to the nearest cluster centroid,
    matching Eq. (16)-Eq. (17) in the manuscript. DBSCAN outliers remain
    unchanged by default.
    """
    if isinstance(confidence, ConfidenceResult):
        result = adaptive_selective_refinement(
            features,
            pseudo_labels,
            confidence,
            outlier_label=outlier_label,
        )
        return result.refined_labels

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
