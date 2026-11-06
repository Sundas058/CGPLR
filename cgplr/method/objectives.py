import torch
from torch import nn
import torch.nn.functional as F

from .refinement import DEFER, PRESERVE, REASSIGN


def _normalizer(weights, eps):
    return weights.sum() + eps


def confidence_weighted_cross_entropy(logits, labels, confidence, eps=1e-12):

    labels = torch.as_tensor(labels, device=logits.device).long()
    confidence = torch.as_tensor(confidence, device=logits.device).float().detach()
    if labels.numel() == 0:
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits, labels, reduction='none')
    return (confidence * losses).sum() / _normalizer(confidence, eps)


def confidence_weighted_triplet_loss(features, labels, confidence, margin=0.3, eps=1e-12):

    features = F.normalize(features, dim=1)
    labels = torch.as_tensor(labels, device=features.device).long()
    confidence = torch.as_tensor(confidence, device=features.device).float().detach()
    distances = torch.cdist(features, features, p=2).square()
    same_id = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positives = same_id & ~eye
    negatives = ~same_id

    valid = positives.any(dim=1) & negatives.any(dim=1)
    if valid.sum() == 0:
        return features.sum() * 0.0

    hardest_positive = distances.masked_fill(~positives, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~negatives, float('inf')).min(dim=1).values
    losses = F.relu(hardest_positive - hardest_negative + margin)
    weights = confidence * valid.float()
    return (weights * losses).sum() / _normalizer(weights, eps)


class ConfidenceAwareTargetLoss(nn.Module):


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


def deferred_consistency_loss(weak_features, strong_features, deferred_mask=None, eps=1e-12):

    weak_features = F.normalize(weak_features, dim=1).detach()
    strong_features = F.normalize(strong_features, dim=1)
    if deferred_mask is not None:
        deferred_mask = torch.as_tensor(deferred_mask, device=strong_features.device).bool()
        if deferred_mask.sum() == 0:
            return strong_features.sum() * 0.0
        weak_features = weak_features[deferred_mask]
        strong_features = strong_features[deferred_mask]
    return ((weak_features - strong_features) ** 2).sum(dim=1).sum() / (
        weak_features.size(0) + eps
    )


class ReliabilityAwareSelectiveLoss(nn.Module):


    def __init__(self, triplet_margin=0.3, triplet_weight=1.0, deferred_weight=0.1, eps=1e-12):
        super(ReliabilityAwareSelectiveLoss, self).__init__()
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.deferred_weight = deferred_weight
        self.eps = eps

    def forward(
        self,
        logits,
        features,
        refined_labels,
        learning_weights,
        decisions,
        weak_features=None,
        strong_features=None,
    ):
        decisions = torch.as_tensor(decisions, device=features.device).long()
        refined_labels = torch.as_tensor(refined_labels, device=features.device).long()
        learning_weights = torch.as_tensor(learning_weights, device=features.device).float()
        hard_mask = (decisions == PRESERVE) | (decisions == REASSIGN)
        if hard_mask.sum() == 0:
            ce_loss = logits.sum() * 0.0
            tri_loss = features.sum() * 0.0
        else:
            ce_loss = confidence_weighted_cross_entropy(
                logits[hard_mask],
                refined_labels[hard_mask],
                learning_weights[hard_mask],
                eps=self.eps,
            )
            tri_loss = confidence_weighted_triplet_loss(
                features[hard_mask],
                refined_labels[hard_mask],
                learning_weights[hard_mask],
                margin=self.triplet_margin,
                eps=self.eps,
            )

        if weak_features is None or strong_features is None:
            def_loss = features.new_tensor(0.0)
        else:
            def_loss = deferred_consistency_loss(
                weak_features,
                strong_features,
                deferred_mask=(decisions == DEFER),
                eps=self.eps,
            )
        return ce_loss + self.triplet_weight * tri_loss + self.deferred_weight * def_loss
