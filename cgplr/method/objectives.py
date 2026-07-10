"""Confidence-aware target-domain objectives for CGPLR."""

import torch
from torch import nn
import torch.nn.functional as F


def _normalizer(weights, eps):
    return weights.sum().clamp_min(eps)


def confidence_weighted_cross_entropy(logits, labels, confidence, eps=1e-12):
    """Compute Eq. (18), confidence-weighted target cross entropy."""
    labels = torch.as_tensor(labels, device=logits.device).long()
    confidence = torch.as_tensor(confidence, device=logits.device).float()
    losses = F.cross_entropy(logits, labels, reduction='none')
    return (confidence * losses).sum() / _normalizer(confidence, eps)


def confidence_weighted_triplet_loss(features, labels, confidence, margin=0.3, eps=1e-12):
    """Compute a batch-hard confidence-weighted triplet loss.

    The anchor confidence weights each triplet term, matching the confidence
    weighting in Eq. (19).
    """
    features = F.normalize(features, dim=1)
    labels = torch.as_tensor(labels, device=features.device).long()
    confidence = torch.as_tensor(confidence, device=features.device).float()
    distances = torch.cdist(features, features, p=2)
    same_id = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positives = same_id & ~eye
    negatives = ~same_id

    valid = positives.any(dim=1) & negatives.any(dim=1)
    if valid.sum() == 0:
        return features.new_tensor(0.0)

    hardest_positive = distances.masked_fill(~positives, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~negatives, float('inf')).min(dim=1).values
    losses = F.relu(hardest_positive - hardest_negative + margin)
    weights = confidence * valid.float()
    return (weights * losses).sum() / _normalizer(weights, eps)


class ConfidenceAwareTargetLoss(nn.Module):
    """Target-domain loss in Eq. (20): weighted CE plus weighted triplet."""

    def __init__(self, triplet_margin=0.3, triplet_weight=1.0, eps=1e-12):
        super(ConfidenceAwareTargetLoss, self).__init__()
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.eps = eps

    def forward(self, logits, features, labels, confidence):
        ce_loss = confidence_weighted_cross_entropy(
            logits,
            labels,
            confidence,
            eps=self.eps,
        )
        tri_loss = confidence_weighted_triplet_loss(
            features,
            labels,
            confidence,
            margin=self.triplet_margin,
            eps=self.eps,
        )
        return ce_loss + self.triplet_weight * tri_loss
