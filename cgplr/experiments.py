ABLATIONS = {
    'full': {},
    'source_only': {},
    'clustering': {'clustering_baseline': True, 'deferred_weight': 0.0},
    'centroid': {'refinement_mode': 'raw', 'alpha': 1.0, 'structural_weight': 1.0,
                 'margin_weight': 0.0, 'temporal_weight': 0.0, 'deferred_weight': 0.0},
    'neighborhood': {'refinement_mode': 'raw', 'alpha': 0.5, 'structural_weight': 1.0,
                     'margin_weight': 0.0, 'temporal_weight': 0.0, 'deferred_weight': 0.0},
    'refinement': {'uniform_weights': True, 'discard_deferred': True},
    'no_margin': {'margin_weight': 0.0},
    'no_temporal': {'temporal_weight': 0.0},
    'global_thresholds': {'global_thresholds': True},
    'forced_reassignment': {'force_reassignment': True},
    'discard_deferred': {'discard_deferred': True},
    'no_consistency': {'deferred_weight': 0.0},
    'equal_weights': {'uniform_weights': True},
    'overlap_history': {'temporal_mode': 'overlap'},
    'candidate_guard': {'min_candidate_size': 8, 'min_candidate_reliability': 0.5},
}


def apply_ablation(args):

    for key, value in ABLATIONS[args.ablation].items():
        setattr(args, key, value)
    if args.ablation == 'source_only' and args.stage != 'evaluate':
        raise ValueError('source_only uses --stage evaluate and a source checkpoint')
    return args


def deferred_batch_mask(deferred, discard=False):

    return deferred & (not discard)
