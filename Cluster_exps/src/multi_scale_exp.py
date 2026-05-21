import os
import time
import argparse
import numpy as np
import logging

from pipeline.io import setup_logging, log_device_info, load_embeddings, prepare_embeddings, save_results
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, build_knn_graph
from pipeline.leiden import run_leiden_multiscale, build_cluster_hierarchy, DEFAULT_RESOLUTIONS
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import get_ground_truth_stats, evaluate_clustering

def run_pipeline(
    *,
    fasta_path,
    emb_path,
    results_path,
    k_coeff=0.6,
    k_cap=150,
    k_override=None,
    resolution_grid=None,
    objective_function="CPM",
    seed=0,
    centroid_cos_threshold=0.85,
    edge_connectivity_threshold=0.05,
    use_gpu=False,
    graph_type="snn",
    rescue_edges=True,
    rescue_cos_threshold=0.92,
    homology_rescue=True,
    homology_rescue_cos=0.95,
    target_cluster_ratio=None,
    max_selected_levels=None,
    logger=None,
):
    """End-to-end multiscale SNN + Leiden pipeline."""
    log = logger or logging.getLogger("multiscale")
    t_total = time.time()

    log.info("=" * 60)
    log.info("Multiscale SNN + Leiden Pipeline")
    device = log_device_info(log, use_gpu=use_gpu)
    log.info(f"  Dataset: {fasta_path}")
    log.info(f"  Graph type: {graph_type}")
    log.info(f"  Layer: 34, PCA: 400")
    log.info(f"  Adaptive k: coeff={k_coeff}, cap={k_cap}" + (f", override={k_override}" if k_override else ""))
    log.info(f"  Resolutions: {resolution_grid or 'Profile-derived'}")
    log.info(f"  Merge: cos>={centroid_cos_threshold}, edge>={edge_connectivity_threshold}")
    log.info(f"  Rescue edges: {rescue_edges} (cos>={rescue_cos_threshold})")
    log.info(f"  Homology rescue: {homology_rescue} (cos>={homology_rescue_cos})")
    log.info("=" * 60)

    # Step 1: Load + prepare
    t0 = time.time()
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=log)
    X_raw = X_by_layer[34]
    gt_stats = get_ground_truth_stats(prot_meta)
    log.info(
        f"  Ground truth: {gt_stats['n_orthogroups']} orthogroups, "
        f"{gt_stats['true_positive_pairs']} same-orthogroup protein pairs"
    )

    prep = prepare_embeddings(X_raw, pca_dim=400, logger=log)
    X = prep["X"]
    X_pca = prep["X_pca"]
    N = prep["N"]
    log.info(f"  Step 1 took {time.time() - t0:.1f}s")

    # Step 2: Graph construction
    t0 = time.time()
    k = k_override if k_override else compute_adaptive_k(N, coeff=k_coeff, cap=k_cap)
    log.info(f"  Adaptive k = {k} (N={N})")

    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=use_gpu, logger=log)
    
    if graph_type == "knn":
        t0 = time.time()
        snn = build_knn_graph(candidates, logger=log)
        log.info(f"  Step 2 (KNN graph) took {time.time() - t0:.1f}s")
    else:
        t0 = time.time()
        snn = build_snn_graph(
            candidates,
            X=X,
            prune_method="inverse_k",
            min_giant_component_pct=90.0,
            rescue_edges=rescue_edges,
            rescue_cos_threshold=rescue_cos_threshold,
            logger=log,
        )
        log.info(f"  Step 2 (SNN graph) took {time.time() - t0:.1f}s")

    # Step 3: Architecture discovery
    if resolution_grid is not None:
        ms_result = run_leiden_multiscale(
            snn,
            resolution_grid=resolution_grid,
            objective_function=objective_function,
            seed=seed,
            min_cluster_size=1,
            logger=log,
        )
        hierarchy = build_cluster_hierarchy(ms_result, logger=log)
        log.info(f"  Step 3 (Leiden Sweeps) took {time.time() - t0:.1f}s")
    else:
        from pipeline.discovery import run_resolution_profile_discovery
        ms_result = run_resolution_profile_discovery(
            snn,
            objective_function=objective_function,
            per_component=True,
            min_component_size=50,
            logger=log,
        )
        hierarchy = build_cluster_hierarchy(ms_result, logger=log)
        log.info(f"  Resolution profile discovery took {time.time() - t0:.1f}s")

    # Step 4: Stability + selection (Always Run)
    t0 = time.time()
    stability = score_and_select_scales(
        hierarchy, ms_result, snn,
        consensus_runs=5,
        seed=seed,
        objective_function=objective_function,
        target_cluster_ratio=target_cluster_ratio,
        max_selected_levels=max_selected_levels,
        logger=log,
    )
    log.info(f"  Step 4 (Scoring) took {time.time() - t0:.1f}s")

    # Step 5: Refinement + flattening
    t0 = time.time()
    refined = refine_and_flatten(
        X_pca, snn, hierarchy, stability, ms_result,
        centroid_cos_threshold=centroid_cos_threshold,
        edge_connectivity_threshold=edge_connectivity_threshold,
        k_neighbors=k,
        homology_rescue=homology_rescue,
        homology_rescue_cos=homology_rescue_cos,
        logger=log,
    )
    log.info(f"  Step 5 took {time.time() - t0:.1f}s")

    # Step 6: Evaluate all levels
    t0 = time.time()
    all_metrics = {}
    for level_name, level_labels in refined["labels_all"].items():
        all_metrics[level_name] = evaluate_clustering(
            level_labels,
            prot_meta,
            level_name=level_name,
            logger=log,
        )
    log.info(f"  Step 6 took {time.time() - t0:.1f}s")

    total_time = time.time() - t_total
    log.info("=" * 60)
    log.info(f"Pipeline complete in {total_time:.1f}s")
    log.info("=" * 60)

    # Compile config
    config = {
        "fasta_path": fasta_path,
        "emb_path": emb_path,
        "layer": 34,
        "pca_dim": 400,
        "k_coeff": k_coeff,
        "k_cap": k_cap,
        "graph_type": graph_type,
        "k_used": k,
        "resolution_grid": resolution_grid,
        "objective_function": objective_function,
        "centroid_cos_threshold": centroid_cos_threshold,
        "edge_connectivity_threshold": edge_connectivity_threshold,
        "seed": seed,
        "use_gpu": use_gpu,
        "device": device,
        "total_time_sec": total_time,
        "rescue_edges": rescue_edges,
        "rescue_cos_threshold": rescue_cos_threshold,
        "homology_rescue": homology_rescue,
        "homology_rescue_cos": homology_rescue_cos,
        "target_cluster_ratio": target_cluster_ratio,
        "max_selected_levels": max_selected_levels,
    }

    return {
        "labels": refined["labels"],
        "labels_all": refined["labels_all"],
        "metrics": all_metrics,
        "hierarchy": hierarchy,
        "stability": stability,
        "graph_stats": snn["stats"],
        "merge_log": refined["merge_log"],
        "n_merges": refined["n_merges"],
        "stage_summaries": refined.get("stage_summaries", []),
        "config": config,
        "prot_meta": prot_meta,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiscale SNN + Leiden clustering pipeline for ortholog detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("organism", type=str, help="Organism code, e.g. pfal_pber")
    parser.add_argument("--k_coeff", type=float, default=0.6, help="k = coeff * sqrt(N) (default: 0.6)")
    parser.add_argument("--k_cap", type=int, default=150, help="Max k (default: 150)")
    parser.add_argument("--k_override", type=int, default=None, help="Override adaptive k with fixed value")
    parser.add_argument("--resolutions", type=str, default=None, help="Comma-separated CPM resolutions (default: None, uses resolution profile)")
    parser.add_argument("--objective", type=str, default="CPM", choices=["CPM", "modularity"], help="Leiden objective")
    parser.add_argument("--centroid_cos_threshold", type=float, default=0.85, help="Centroid cosine merge threshold (default: 0.85)")
    parser.add_argument("--edge_connectivity_threshold", type=float, default=0.05, help="SNN edge connectivity merge threshold (default: 0.05)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration when available (FAISS GPU, PyTorch CUDA).")
    parser.add_argument("--graph_type", type=str, default="snn", choices=["snn", "knn"], help="Graph architecture to build (snn or knn)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # ── New recall-oriented flags ──
    parser.add_argument("--no_rescue_edges", action="store_true", help="Disable hybrid rescue edges in graph builder (default: on)")
    parser.add_argument("--rescue_cos_threshold", type=float, default=0.92, help="Cosine threshold for rescue edge eligibility (default: 0.92)")
    parser.add_argument("--no_homology_rescue", action="store_true", help="Disable homology-rescue merge path")
    parser.add_argument("--homology_rescue_cos", type=float, default=0.95, help="Cosine threshold for homology-rescue merges (default: 0.95)")
    parser.add_argument("--target_cluster_ratio", type=float, default=None, help="Soft prior for target K as fraction of N (default: 1/3)")
    parser.add_argument("--max_selected_levels", type=int, default=None, help="Max hierarchy levels to select (default: 6)")

    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = "../OrthoMCL/"
    results_root = "../Model_Cluster_Results/multiscale"
    species = f"{args.organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{args.organism}.fasta")
    emb_path = os.path.join(dataset_path, species)
    results_path = os.path.join(results_root, species)

    resolution_grid = None
    if args.resolutions:
        resolution_grid = [float(r.strip()) for r in args.resolutions.split(",")]

    organism_name = os.path.splitext(os.path.basename(fasta_path))[0]
    logger = setup_logging(organism_name)
    import sys as _sys
    logger.info(f"CMD: {' '.join(_sys.argv)}")
    np.random.seed(args.seed)

    results = run_pipeline(
        fasta_path=fasta_path,
        emb_path=emb_path,
        results_path=results_path,
        k_coeff=args.k_coeff,
        k_cap=args.k_cap,
        k_override=args.k_override,
        resolution_grid=resolution_grid,
        objective_function=args.objective,
        seed=args.seed,
        centroid_cos_threshold=args.centroid_cos_threshold,
        edge_connectivity_threshold=args.edge_connectivity_threshold,
        use_gpu=args.use_gpu,
        graph_type=args.graph_type,
        rescue_edges=not args.no_rescue_edges,
        rescue_cos_threshold=args.rescue_cos_threshold,
        homology_rescue=not args.no_homology_rescue,
        homology_rescue_cos=args.homology_rescue_cos,
        target_cluster_ratio=args.target_cluster_ratio,
        max_selected_levels=args.max_selected_levels,
        logger=logger,
    )
    save_results(results, results_path, args.organism, logger=logger)

if __name__ == "__main__":
    main()
