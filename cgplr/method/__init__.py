"""Method components for CGPLR.

The modules in this package mirror Section 3 of the manuscript:
source-domain initialization, target-domain clustering, confidence
estimation, pseudo-label refinement, and confidence-aware learning.
"""

from .source_training import SourceInitializationLoss, source_reid_loss
from .clustering import cluster_with_dbscan, compute_cluster_centroids, valid_cluster_mask
from .confidence import estimate_sample_confidence
from .refinement import partition_by_confidence, refine_pseudo_labels
from .objectives import (
    ConfidenceAwareTargetLoss,
    confidence_weighted_cross_entropy,
    confidence_weighted_triplet_loss,
)

__all__ = [
    'SourceInitializationLoss',
    'source_reid_loss',
    'cluster_with_dbscan',
    'compute_cluster_centroids',
    'valid_cluster_mask',
    'estimate_sample_confidence',
    'partition_by_confidence',
    'refine_pseudo_labels',
    'ConfidenceAwareTargetLoss',
    'confidence_weighted_cross_entropy',
    'confidence_weighted_triplet_loss',
]
