import os
import time
import argparse
import numpy as np
import logging

from pipeline.io import setup_logging, log_device_info, load_embeddings, prepare_embeddings, save_results
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, run_phate_scale_discovery, map_phate_to_leiden_resolutions
from pipeline.leiden import run_leiden_multiscale, build_cluster_hierarchy, DEFAULT_RESOLUTIONS
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import get_ground_truth_stats, evaluate_clustering

def run_pipeline(
    *,
    fasta_path,
    emb_path,
    results_path,
    layer=34,
    pca_dim=400,
    k_coeff=0.6,
    k_cap=150,
    k_override=None,
    prune_method="inverse_k",
    resolution_grid=None,
    objective_function="CPM",
    seed=0,
    centroid_cos_threshold=0.85,
    edge_connectivity_threshold=0.05,
    output_level="fine",
    alpha=0.5,
    min_cluster_size=1,
    selection_policy="best_composite",
    use_phate=False,
    no_pygen=False,
    use_profile=False,
    use_gpu=False,
    # ── New recall-oriented parameters ──
    rescue_edges=True,
    rescue_cos_threshold=0.92,
    homology_rescue=False,
    homology_rescue_cos=0.95,
    cross_branch_rescue=False,
    cross_branch_cos=0.93,
    cross_branch_edge=0.03,
    max_cross_branch_merges=500,
    target_cluster_ratio=None,
    max_selected_levels=None,
    extended_eval=False,
    per_component_discovery=True,
    min_component_size=50,
    logger=None,
):
    """End-to-end multiscale SNN + Leiden pipeline."""
    log = logger or logging.getLogger("multiscale")
    t_total = time.time()

    log.info("=" * 60)
    log.info("Multiscale SNN + Leiden Pipeline")
    device = log_device_info(log, use_gpu=use_gpu)
    log.info(f"  Dataset: {fasta_path}")
    log.info(f"  Layer: {layer}, PCA: {pca_dim}")
    log.info(f"  Adaptive k: coeff={k_coeff}, cap={k_cap}" + (f", override={k_override}" if k_override else ""))
    log.info(f"  PHATE scale discovery: {use_phate}")
    log.info(f"  Resolutions: {resolution_grid or ('PHATE-derived' if use_phate else 'Default')}")
    log.info(f"  Merge: cos>={centroid_cos_threshold}, edge>={edge_connectivity_threshold}")
    log.info(f"  Rescue edges: {rescue_edges} (cos>={rescue_cos_threshold})")
    log.info(f"  Homology rescue: {homology_rescue} (cos>={homology_rescue_cos})")
    log.info(f"  Cross-branch rescue: {cross_branch_rescue}")
    log.info(f"  Extended eval: {extended_eval}")
    log.info(f"  Output level: {output_level}")
    log.info("=" * 60)
    if alpha != 0.5:
        log.warning("  --alpha is deprecated and has no effect on the pairwise evaluator.")
    if min_cluster_size != 1:
        log.warning("  --min_cluster_size is currently not enforced in Leiden outputs and is recorded for traceability only.")

    # Step 1: Load + prepare
    t0 = time.time()
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [layer], logger=log)
    X_raw = X_by_layer[layer]
    gt_stats = get_ground_truth_stats(prot_meta)
    log.info(
        f"  Ground truth: {gt_stats['n_orthogroups']} orthogroups, "
        f"{gt_stats['true_positive_pairs']} same-orthogroup protein pairs"
    )

    prep = prepare_embeddings(X_raw, pca_dim=pca_dim, logger=log)
    X = prep["X"]
    X_pca = prep["X_pca"]
    N = prep["N"]
    log.info(f"  Step 1 took {time.time() - t0:.1f}s")

    # Step 2: SNN graph
    t0 = time.time()
    k = k_override if k_override else compute_adaptive_k(N, coeff=k_coeff, cap=k_cap)
    log.info(f"  Adaptive k = {k} (N={N})")

    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=use_gpu, logger=log)
    snn = build_snn_graph(
        candidates,
        X=X,
        prune_method=prune_method,
        rescue_edges=rescue_edges,
        rescue_cos_threshold=rescue_cos_threshold,
        logger=log,
    )
    log.info(f"  Step 2 took {time.time() - t0:.1f}s")

    mode = "pygen"
    if use_phate:
        mode = "phate"
    elif use_profile:
        mode = "profile"
    elif no_pygen:
        mode = "manual"

    phate_result = None
    t0 = time.time()
    
    # Step 3: Architecture discovery
    if mode == "pygen":
        from pipeline.discovery import run_pygenstability_discovery
        ms_result = run_pygenstability_discovery(
            snn,
            per_component=per_component_discovery,
            min_component_size=min_component_size,
            logger=log,
        )
        
        # PyGen Fallback Rule
        if len(ms_result["levels"]) < 3:
            log.warning("  [FALLBACK] PyGen → manual grid: returned < 3 feasible scales.")
            if resolution_grid is None:
                resolution_grid = DEFAULT_RESOLUTIONS
            ms_result = run_leiden_multiscale(
                snn,
                resolution_grid=resolution_grid,
                objective_function=objective_function,
                seed=seed,
                min_cluster_size=min_cluster_size,
                logger=log,
            )
            mode = "pygen_fallback_manual"
            
        hierarchy = build_cluster_hierarchy(ms_result, logger=log)
        log.info(f"  PyGenStability discovery took {time.time() - t0:.1f}s")
        
    elif mode == "profile":
        from pipeline.discovery import run_resolution_profile_discovery
        ms_result = run_resolution_profile_discovery(
            snn,
            objective_function=objective_function,
            per_component=per_component_discovery,
            min_component_size=min_component_size,
            logger=log,
        )
        hierarchy = build_cluster_hierarchy(ms_result, logger=log)
        log.info(f"  Resolution profile discovery took {time.time() - t0:.1f}s")
        
    elif mode in ["phate", "manual"]:
        if mode == "phate" and resolution_grid is None:
            phate_result = run_phate_scale_discovery(X_pca, random_state=seed, logger=log)
            if len(phate_result["cluster_counts"]) >= 2:
                resolution_grid = map_phate_to_leiden_resolutions(
                    snn, phate_result["cluster_counts"],
                    objective=objective_function, logger=log,
                )
                log.info(f"  PHATE scale discovery took {time.time() - t0:.1f}s")
            else:
                log.warning("  PHATE returned <2 levels, falling back to default grid.")
                phate_result = None
        elif mode == "phate" and resolution_grid is not None:
            log.info("  --use_phate ignored because --resolutions was explicitly set.")
    
        ms_result = run_leiden_multiscale(
            snn,
            resolution_grid=resolution_grid,
            objective_function=objective_function,
            seed=seed,
            min_cluster_size=min_cluster_size,
            logger=log,
        )
        hierarchy = build_cluster_hierarchy(ms_result, logger=log)
        log.info(f"  Step 3 (Leiden Sweeps) took {time.time() - t0:.1f}s")

    # Step 4: Stability + selection (Always Run)
    t0 = time.time()
    stability = score_and_select_scales(
        hierarchy, ms_result, snn,
        selection_policy=selection_policy,
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
        output_level=output_level,
        k_neighbors=k,
        homology_rescue=homology_rescue,
        homology_rescue_cos=homology_rescue_cos,
        cross_branch_rescue=cross_branch_rescue,
        cross_branch_cos=cross_branch_cos,
        cross_branch_edge=cross_branch_edge,
        max_cross_branch_merges=max_cross_branch_merges,
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
            graph=snn.get("graph"),
            level_name=level_name,
            extended_eval=extended_eval,
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
        "layer": layer,
        "pca_dim": pca_dim,
        "k_coeff": k_coeff,
        "k_cap": k_cap,
        "k_used": k,
        "prune_method": prune_method,
        "resolution_grid": resolution_grid,
        "objective_function": objective_function,
        "centroid_cos_threshold": centroid_cos_threshold,
        "edge_connectivity_threshold": edge_connectivity_threshold,
        "output_level": output_level,
        "alpha_deprecated": alpha,
        "seed": seed,
        "min_cluster_size": min_cluster_size,
        "discovery_mode": mode,
        "use_phate": use_phate,
        "no_pygen": no_pygen,
        "use_profile": use_profile,
        "use_gpu": use_gpu,
        "device": device,
        "phate_levels": phate_result["cluster_counts"] if phate_result else None,
        "total_time_sec": total_time,
        # New config entries
        "rescue_edges": rescue_edges,
        "rescue_cos_threshold": rescue_cos_threshold,
        "homology_rescue": homology_rescue,
        "homology_rescue_cos": homology_rescue_cos,
        "cross_branch_rescue": cross_branch_rescue,
        "cross_branch_cos": cross_branch_cos,
        "cross_branch_edge": cross_branch_edge,
        "target_cluster_ratio": target_cluster_ratio,
        "max_selected_levels": max_selected_levels,
        "extended_eval": extended_eval,
        "per_component_discovery": per_component_discovery,
        "min_component_size": min_component_size,
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
    parser.add_argument("--layer", type=int, default=34, help="ESM-C layer (default: 34)")
    parser.add_argument("--pca_dim", type=int, default=400, help="PCA components (default: 400)")
    parser.add_argument("--k_coeff", type=float, default=0.6, help="k = coeff * sqrt(N) (default: 0.6)")
    parser.add_argument("--k_cap", type=int, default=150, help="Max k (default: 150)")
    parser.add_argument("--k_override", type=int, default=None, help="Override adaptive k with fixed value")
    parser.add_argument("--prune_method", type=str, default="inverse_k", choices=["inverse_k", "percentile"], help="SNN pruning method")
    parser.add_argument("--resolutions", type=str, default=None, help="Comma-separated CPM resolutions (default: 0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.45,0.60,0.80,1.00)")
    parser.add_argument("--objective", type=str, default="CPM", choices=["CPM", "modularity"], help="Leiden objective")
    parser.add_argument("--min_cluster_size", type=int, default=1, help="Recorded for traceability; currently not enforced in Leiden outputs (default: 1)")
    parser.add_argument("--centroid_cos_threshold", type=float, default=0.85, help="Centroid cosine merge threshold (default: 0.85)")
    parser.add_argument("--edge_connectivity_threshold", type=float, default=0.05, help="SNN edge connectivity merge threshold (default: 0.05)")
    parser.add_argument("--output_level", type=str, default="fine", choices=["coarse", "fine", "adaptive"], help="Primary output level")
    parser.add_argument("--alpha", type=float, default=0.5, help="Deprecated: retained for CLI compatibility, no effect on the pairwise evaluator")
    parser.add_argument("--use_phate", action="store_true", help="Use Multiscale PHATE for automatic scale discovery")
    parser.add_argument("--no_pygen", action="store_true", help="Disable default PyGenStability and use manual heuristic grid")
    parser.add_argument("--use_profile", action="store_true", help="Use leidenalg.resolution_profile for scale discovery")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration when available (FAISS GPU, PyTorch CUDA).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--selection_policy", type=str, default="best_composite", choices=["best_composite", "max_stability", "elbow"], help="Scale selection policy")

    # ── New recall-oriented flags ──
    parser.add_argument("--no_rescue_edges", action="store_true", help="Disable hybrid rescue edges in graph builder (default: on)")
    parser.add_argument("--rescue_cos_threshold", type=float, default=0.92, help="Cosine threshold for rescue edge eligibility (default: 0.92)")
    parser.add_argument("--homology_rescue", action="store_true", help="Enable homology-rescue merge path for small clusters with very high cosine")
    parser.add_argument("--homology_rescue_cos", type=float, default=0.95, help="Cosine threshold for homology-rescue merges (default: 0.95)")
    parser.add_argument("--cross_branch_rescue", action="store_true", help="Enable cross-branch merge rescue after cascade")
    parser.add_argument("--cross_branch_cos", type=float, default=0.93, help="Cosine threshold for cross-branch rescue (default: 0.93)")
    parser.add_argument("--cross_branch_edge", type=float, default=0.03, help="Edge connectivity threshold for cross-branch rescue (default: 0.03)")
    parser.add_argument("--max_cross_branch_merges", type=int, default=500, help="Max cross-branch merges per pass (default: 500)")
    parser.add_argument("--target_cluster_ratio", type=float, default=None, help="Soft prior for target K as fraction of N (default: 1/3)")
    parser.add_argument("--max_selected_levels", type=int, default=None, help="Max hierarchy levels to select (default: 6)")
    parser.add_argument("--extended_eval", action="store_true", help="Enable size-binned, B-cubed, and split/merge evaluation")
    parser.add_argument("--no_per_component_discovery", action="store_true", help="Disable per-component PyGen/profile discovery (default: on)")
    parser.add_argument("--min_component_size", type=int, default=50, help="Min component size for per-component discovery (default: 50)")

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
        layer=args.layer,
        pca_dim=args.pca_dim,
        k_coeff=args.k_coeff,
        k_cap=args.k_cap,
        k_override=args.k_override,
        prune_method=args.prune_method,
        resolution_grid=resolution_grid,
        objective_function=args.objective,
        seed=args.seed,
        centroid_cos_threshold=args.centroid_cos_threshold,
        edge_connectivity_threshold=args.edge_connectivity_threshold,
        output_level=args.output_level,
        alpha=args.alpha,
        min_cluster_size=args.min_cluster_size,
        selection_policy=args.selection_policy,
        use_phate=args.use_phate,
        no_pygen=args.no_pygen,
        use_profile=args.use_profile,
        use_gpu=args.use_gpu,
        # New parameters
        rescue_edges=not args.no_rescue_edges,
        rescue_cos_threshold=args.rescue_cos_threshold,
        homology_rescue=args.homology_rescue,
        homology_rescue_cos=args.homology_rescue_cos,
        cross_branch_rescue=args.cross_branch_rescue,
        cross_branch_cos=args.cross_branch_cos,
        cross_branch_edge=args.cross_branch_edge,
        max_cross_branch_merges=args.max_cross_branch_merges,
        target_cluster_ratio=args.target_cluster_ratio,
        max_selected_levels=args.max_selected_levels,
        extended_eval=args.extended_eval,
        per_component_discovery=not args.no_per_component_discovery,
        min_component_size=args.min_component_size,
        logger=logger,
    )
    save_results(results, results_path, args.organism, logger=logger)

if __name__ == "__main__":
    main()
