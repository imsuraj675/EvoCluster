import os
import time
import argparse
import numpy as np
import logging

from pipeline.io import setup_logging, log_device_info, load_embeddings, prepare_embeddings, save_results
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, run_phate_scale_discovery, map_phate_to_leiden_resolutions
from pipeline.leiden import run_leiden_multiscale, build_cluster_hierarchy
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import get_total_counts, evaluate_clustering

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
    log.info(f"  Output level: {output_level}, alpha={alpha}")
    log.info("=" * 60)

    # Step 1: Load + prepare
    t0 = time.time()
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [layer], logger=log)
    X_raw = X_by_layer[layer]
    total_n2m, total_121 = get_total_counts(prot_meta)
    log.info(f"  True n:m pairs: {total_n2m}, True 1:1 groups: {total_121}")

    prep = prepare_embeddings(X_raw, pca_dim=pca_dim, logger=log)
    X = prep["X"]
    N = prep["N"]
    # Keep unscaled PCA for evaluation (centroid distances)
    if pca_dim > 0 and pca_dim < X_raw.shape[1]:
        from sklearn.decomposition import PCA
        X_pca = PCA(n_components=pca_dim, svd_solver="full").fit_transform(X_raw)
    else:
        X_pca = X_raw.copy()
    log.info(f"  Step 1 took {time.time() - t0:.1f}s")

    # Step 2: SNN graph
    t0 = time.time()
    k = k_override if k_override else compute_adaptive_k(N, coeff=k_coeff, cap=k_cap)
    log.info(f"  Adaptive k = {k} (N={N})")

    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=use_gpu, logger=log)
    snn = build_snn_graph(candidates, prune_method=prune_method, logger=log)
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
        ms_result = run_pygenstability_discovery(snn, logger=log)
        
        # TASK A: PyGen Fallback Rule
        if len(ms_result["levels"]) < 3:
            log.warning("  ⚠ PyGen returned < 3 feasible scales. Falling back to default Leiden multiscale grid.")
            if resolution_grid is None:
                resolution_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
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
        ms_result = run_resolution_profile_discovery(snn, objective_function=objective_function, logger=log)
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
        logger=log,
    )
    log.info(f"  Step 5 took {time.time() - t0:.1f}s")

    # Step 6: Evaluate all levels
    t0 = time.time()
    all_metrics = {}
    for level_name, level_labels in refined["labels_all"].items():
        all_metrics[level_name] = evaluate_clustering(
            level_labels, prot_meta, X_pca, total_n2m, total_121,
            graph=snn.get("graph"), alpha=alpha, level_name=level_name, logger=log,
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
        "alpha": alpha,
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
    parser.add_argument("--resolutions", type=str, default=None, help="Comma-separated CPM resolutions (default: 0.05,0.1,0.2,0.3,0.5,0.7)")
    parser.add_argument("--objective", type=str, default="CPM", choices=["CPM", "modularity"], help="Leiden objective")
    parser.add_argument("--min_cluster_size", type=int, default=1, help="Min cluster size (default: 1)")
    parser.add_argument("--centroid_cos_threshold", type=float, default=0.85, help="Centroid cosine merge threshold (default: 0.85)")
    parser.add_argument("--edge_connectivity_threshold", type=float, default=0.05, help="SNN edge connectivity merge threshold (default: 0.05)")
    parser.add_argument("--output_level", type=str, default="fine", choices=["coarse", "fine", "adaptive"], help="Primary output level")
    parser.add_argument("--alpha", type=float, default=0.5, help="Combined score weight: alpha*dist_F1 + (1-alpha)*121_F1")
    parser.add_argument("--use_phate", action="store_true", help="Use Multiscale PHATE for automatic scale discovery")
    parser.add_argument("--no_pygen", action="store_true", help="Disable default PyGenStability and use manual heuristic grid")
    parser.add_argument("--use_profile", action="store_true", help="Use leidenalg.resolution_profile for scale discovery")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration when available (FAISS GPU, PyTorch CUDA).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--selection_policy", type=str, default="best_composite", choices=["best_composite", "max_stability", "best_density", "elbow"], help="Scale selection policy")
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
        logger=logger,
    )
    save_results(results, results_path, args.organism, logger=logger)

if __name__ == "__main__":
    main()
