"""
EvoCluster Pipeline Package
"""
from .io import load_embeddings, prepare_embeddings, setup_logging, save_results
from .graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, build_knn_graph
from .leiden import run_leiden_multiscale, build_cluster_hierarchy
from .selection import score_and_select_scales
from .merge import refine_and_flatten
from .evaluation import (
    evaluate_clustering, get_ground_truth_stats, compute_pairwise_confusion_metrics,
    extract_species_labels, compute_orthogroup_type_metrics,
)
from .discovery import run_resolution_profile_discovery
# Core pipeline modules
