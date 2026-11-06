import torch
from torch import nn
import torch.nn.functional as F

from .objectives import confidence_weighted_triplet_loss


def source_reid_loss(logits, features, labels, triplet_margin=0.3, beta=1.0):

    ce_loss = F.cross_entropy(logits, labels)
    tri_loss = confidence_weighted_triplet_loss(features, labels,
                                               features.new_ones(len(labels)), margin=triplet_margin)
    return ce_loss + beta * tri_loss


class SourceInitializationLoss(nn.Module):


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
