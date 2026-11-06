from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .clustering import compute_cluster_centroids, valid_cluster_mask
from .history import temporal_assignment_stability
from .neighbors import nearest_neighbors


@dataclass
class ConfidenceResult:


    weights: torch.Tensor
    structural_scores: torch.Tensor
    centroid_scores: torch.Tensor
    neighborhood_scores: torch.Tensor
    margin_scores: torch.Tensor
    temporal_scores: torch.Tensor
    assigned_distances: torch.Tensor
    competitor_distances: torch.Tensor
    best_competitors: torch.Tensor
    valid_mask: torch.Tensor
    cluster_ids: torch.Tensor
    centroids: torch.Tensor
    neighbor_indices: torch.Tensor


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


def _nearest_neighbors(features, neighbors):
    return nearest_neighbors(features, neighbors)


def _neighborhood_consistency(features, labels, neighbors):
    neighbor_indices = _nearest_neighbors(features, neighbors)
    if neighbor_indices.numel() == 0:
        return features.new_zeros(labels.size(0)), neighbor_indices
    neighbor_labels = labels[neighbor_indices]
    scores = neighbor_labels.eq(labels.view(-1, 1)).float().mean(dim=1)
    return scores, neighbor_indices


def _margin_confidence(features, labels, cluster_ids, centroids, temperature):
    if temperature <= 0:
        raise ValueError('margin temperature must be positive')

    assigned_distances = features.new_full((labels.size(0),), float('inf'))
    competitor_distances = features.new_full((labels.size(0),), float('inf'))
    best_competitors = labels.new_full((labels.size(0),), -1)

    if centroids.size(0) == 0:
        return (
            features.new_zeros(labels.size(0)),
            assigned_distances,
            competitor_distances,
            best_competitors,
        )

    lookup = _centroid_lookup(cluster_ids)
    distances = torch.cdist(features, centroids, p=2) ** 2
    for sample_index, label in enumerate(labels.tolist()):
        assigned_index = lookup.get(int(label))
        if assigned_index is None:
            continue
        assigned_distances[sample_index] = distances[sample_index, assigned_index]
        if centroids.size(0) == 1:
            competitor_distances[sample_index] = assigned_distances[sample_index]
            best_competitors[sample_index] = int(label)
            continue
        candidate_distances = distances[sample_index].clone()
        candidate_distances[assigned_index] = float('inf')
        competitor_index = candidate_distances.argmin()
        competitor_distances[sample_index] = candidate_distances[competitor_index]
        best_competitors[sample_index] = cluster_ids[competitor_index]

    margins = competitor_distances - assigned_distances
    scores = torch.sigmoid(margins / temperature)
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    return scores.clamp_(0.0, 1.0), assigned_distances, competitor_distances, best_competitors


def estimate_sample_confidence(
    features,
    pseudo_labels,
    alpha=0.5,
    temperature=None,
    centroid_temperature=0.5,
    margin_temperature=0.1,
    neighbors=5,
    fusion_weights=(0.5, 0.3, 0.2),
    history=None,
    outlier_label=-1,
    return_details=False,
    temporal_mode='centroid',
):

    if temperature is not None:
        centroid_temperature = temperature
    if centroid_temperature <= 0:
        raise ValueError('centroid temperature must be positive')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    if neighbors < 1:
        raise ValueError('neighbors must be positive')
    if len(fusion_weights) != 3:
        raise ValueError('fusion_weights must contain structural, margin and temporal weights')
    fusion = torch.tensor(fusion_weights, dtype=torch.float64)
    if not torch.isfinite(fusion).all() or (fusion < 0).any() or fusion[:2].sum() <= 0:
        raise ValueError('finite nonnegative fusion weights and positive structural/margin sum required')
    if torch.as_tensor(pseudo_labels).shape != (len(features),):
        raise ValueError('one pseudo label is required per feature')

    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels, device=features.device).long()
    valid_mask = valid_cluster_mask(labels, outlier_label=outlier_label).to(features.device)
    confidence = features.new_zeros(labels.size(0))
    structural = features.new_zeros(labels.size(0))
    centroid_scores = features.new_zeros(labels.size(0))
    neighborhood_scores = features.new_zeros(labels.size(0))
    margin_scores = features.new_zeros(labels.size(0))
    temporal_scores = features.new_zeros(labels.size(0))
    assigned_distances = features.new_full((labels.size(0),), float('inf'))
    competitor_distances = features.new_full((labels.size(0),), float('inf'))
    best_competitors = labels.new_full((labels.size(0),), outlier_label)
    neighbor_indices = torch.empty((0, 0), dtype=torch.long, device=features.device)

    if valid_mask.sum() == 0:
        result = ConfidenceResult(
            weights=confidence,
            structural_scores=structural,
            centroid_scores=centroid_scores,
            neighborhood_scores=neighborhood_scores,
            margin_scores=margin_scores,
            temporal_scores=temporal_scores,
            assigned_distances=assigned_distances,
            competitor_distances=competitor_distances,
            best_competitors=best_competitors,
            valid_mask=valid_mask,
            cluster_ids=torch.empty(0, dtype=torch.long, device=features.device),
            centroids=features.new_empty((0, features.size(1))),
            neighbor_indices=neighbor_indices,
        )
        return result if return_details else confidence

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
        centroid_temperature,
    )
    neighbor_scores, neighbor_indices = _neighborhood_consistency(valid_features, valid_labels, neighbors)
    structural_scores = alpha * centroid_scores + (1.0 - alpha) * neighbor_scores
    margin_valid, assigned_valid, competitor_valid, competitors_valid = _margin_confidence(
        valid_features,
        valid_labels,
        cluster_ids.to(features.device),
        centroids.to(features.device),
        margin_temperature,
    )
    temporal_valid = temporal_assignment_stability(
        labels.detach().cpu(),
        cluster_ids.detach().cpu(),
        centroids.detach().cpu(),
        history,
        outlier_label=outlier_label,
        mode=temporal_mode,
    ).to(features.device)[valid_mask]

    structural[valid_mask] = structural_scores
    centroid_scores_full = features.new_zeros(labels.size(0))
    neighborhood_scores_full = features.new_zeros(labels.size(0))
    centroid_scores_full[valid_mask] = centroid_scores
    neighborhood_scores_full[valid_mask] = neighbor_scores
    margin_scores[valid_mask] = margin_valid
    temporal_scores[valid_mask] = temporal_valid
    assigned_distances[valid_mask] = assigned_valid
    competitor_distances[valid_mask] = competitor_valid
    best_competitors[valid_mask] = competitors_valid

    structural_weight, margin_weight, temporal_weight = [float(weight) for weight in fusion_weights]
    if structural_weight < 0 or margin_weight < 0 or temporal_weight < 0:
        raise ValueError('fusion weights must be non-negative')
    if structural_weight + margin_weight <= 0:
        raise ValueError('structural and margin weights cannot both be zero')

    if history is None or len(history) == 0 or temporal_weight == 0:
        normalizer = structural_weight + margin_weight
        confidence[valid_mask] = (
            structural_weight * structural_scores + margin_weight * margin_valid
        ) / normalizer
    else:
        total = structural_weight + margin_weight + temporal_weight
        confidence[valid_mask] = (
            structural_weight * structural_scores
            + margin_weight * margin_valid
            + temporal_weight * temporal_valid
        ) / max(total, 1e-12)

    result = ConfidenceResult(
        weights=confidence.clamp_(0.0, 1.0),
        structural_scores=structural.clamp_(0.0, 1.0),
        centroid_scores=centroid_scores_full.clamp_(0.0, 1.0),
        neighborhood_scores=neighborhood_scores_full.clamp_(0.0, 1.0),
        margin_scores=margin_scores.clamp_(0.0, 1.0),
        temporal_scores=temporal_scores.clamp_(0.0, 1.0),
        assigned_distances=assigned_distances,
        competitor_distances=competitor_distances,
        best_competitors=best_competitors,
        valid_mask=valid_mask,
        cluster_ids=cluster_ids.to(features.device),
        centroids=centroids.to(features.device),
        neighbor_indices=neighbor_indices,
    )
    return result if return_details else result.weights
