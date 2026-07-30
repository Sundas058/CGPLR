"""Source-domain initialization losses for CGPLR.

This module covers Eq. (7)-Eq. (9) in the manuscript: identity
classification plus triplet-based metric learning on labeled source data.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..models.triplet import TripletLoss


def source_reid_loss(logits, features, labels, triplet_margin=0.3, beta=1.0):
    """Compute the supervised source-domain Re-ID objective.

    Args:
        logits: Classifier outputs with shape ``[batch, num_source_ids]``.
        features: Embeddings with shape ``[batch, feature_dim]``.
        labels: Source identity labels with shape ``[batch]``.
        triplet_margin: Margin used by the batch-hard triplet loss.
        beta: Weight for the triplet term in Eq. (9).
    """
    ce_loss = F.cross_entropy(logits, labels)
    tri_loss = TripletLoss(margin=triplet_margin)(features, labels)
    return ce_loss + beta * tri_loss


class SourceInitializationLoss(nn.Module):
    """Reusable source-domain objective used before target adaptation."""

    def __init__(self, triplet_margin=0.3, beta=1.0):
        super(SourceInitializationLoss, self).__init__()
        self.triplet_margin = triplet_margin
        self.beta = beta

    def forward(self, logits, features, labels):
        return source_reid_loss(
            logits=logits,
            features=features,
            labels=labels,
            triplet_margin=self.triplet_margin,
            beta=self.beta,
        )
