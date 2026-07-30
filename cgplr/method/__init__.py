"""Method components for CGPLR.

The modules in this package mirror Section 3 of the manuscript:
source-domain initialization, target-domain clustering, confidence
estimation, pseudo-label refinement, and confidence-aware learning.
"""

from .source_training import SourceInitializationLoss, source_reid_loss
from .clustering import (
    cluster_with_dbscan,
    compact_cluster_labels,
    compute_cluster_centroids,
    valid_cluster_mask,
)
from .confidence import ConfidenceResult, estimate_sample_confidence
from .history import ClusterHistory, ClusterSnapshot
from .refinement import (
    DEFER,
    PRESERVE,
    REASSIGN,
    RefinementResult,
    adaptive_selective_refinement,
    assignment_evidence,
    cluster_adaptive_thresholds,
    partition_by_confidence,
    refine_pseudo_labels,
)
from .objectives import (
    ConfidenceAwareTargetLoss,
    ReliabilityAwareSelectiveLoss,
    confidence_weighted_cross_entropy,
    confidence_weighted_triplet_loss,
    deferred_consistency_loss,
)

__all__ = [
    'SourceInitializationLoss',
    'source_reid_loss',
    'cluster_with_dbscan',
    'compact_cluster_labels',
    'compute_cluster_centroids',
    'valid_cluster_mask',
    'ClusterHistory',
    'ClusterSnapshot',
    'ConfidenceResult',
    'estimate_sample_confidence',
    'PRESERVE',
    'REASSIGN',
    'DEFER',
    'RefinementResult',
    'adaptive_selective_refinement',
    'assignment_evidence',
    'cluster_adaptive_thresholds',
    'partition_by_confidence',
    'refine_pseudo_labels',
    'ConfidenceAwareTargetLoss',
    'ReliabilityAwareSelectiveLoss',
    'confidence_weighted_cross_entropy',
    'confidence_weighted_triplet_loss',
    'deferred_consistency_loss',
]
