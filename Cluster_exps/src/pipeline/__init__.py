"""
EvoCluster Pipeline Package
"""
from .io import load_embeddings, prepare_embeddings, setup_logging, save_results
from .graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, run_phate_scale_discovery, map_phate_to_leiden_resolutions
from .leiden import run_leiden_multiscale, build_cluster_hierarchy
from .selection import score_and_select_scales
from .merge import refine_and_flatten
from .evaluation import evaluate_clustering, get_ground_truth_stats, compute_pairwise_confusion_metrics
from .evaluation_extended import (
    run_extended_evaluation,
    compute_size_binned_metrics,
    compute_bcubed_metrics,
    compute_split_merge_errors,
    classify_family_types,
)
from .discovery import run_pygenstability_discovery, run_resolution_profile_discovery
# Core pipeline modules
