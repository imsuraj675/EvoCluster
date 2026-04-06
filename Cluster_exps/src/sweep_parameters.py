import os
import time
import argparse
import numpy as np
import logging
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from pipeline.io import setup_logging, load_embeddings, prepare_embeddings
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph
from pipeline.leiden import build_cluster_hierarchy
from pipeline.discovery import run_resolution_profile_discovery
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import evaluate_clustering

def plot_nested_results(all_studies_data, organism):
    if not all_studies_data:
        return
    
    k_coeffs = []
    best_f1s = []
    
    for data in all_studies_data:
        k_coeffs.append(data["k_coeff"])
        best_f1s.append(data["best_value"])
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_coeffs, best_f1s, marker='o', linestyle='-', color='indigo', linewidth=2)
    plt.title("Best Pairwise F1 vs Adaptive Graph 'k_coeff'")
    plt.xlabel("k_coeff (Graph Sparsity)")
    plt.ylabel("Optimal Pairwise F1 Achieved")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_img = f"sweep_nested_{organism}_kcoeff_performance.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    
    combined = []
    for data in all_studies_data:
        df = data["df"]
        df["k_coeff_used"] = data["k_coeff"]
        combined.append(df)
        
    if combined:
        final_df = pd.concat(combined, ignore_index=True)
        final_df.to_csv(f"sweep_nested_{organism}_full_results.csv", index=False)
        print(f"\n[+] Saved global optimization metric plot to: {out_img}")
        print(f"[+] Saved complete trial log to: sweep_nested_{organism}_full_results.csv")


def optimize_for_k(k_coeff, organism, n_inner_trials, X, X_pca, N, prot_meta):
    # Setup isolated logger for parallel process output safety
    logger_name = f"{organism}_sweep_k{k_coeff}"
    logger = setup_logging(logger_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("\n" + "=" * 60)
    logger.info(f"OUTER SWEEP WORKER STARTING: Building graph for k_coeff = {k_coeff}")
    logger.info("=" * 60)
    
    t_graph = time.time()
    
    k = compute_adaptive_k(N, coeff=k_coeff, cap=150)
    logger.info(f"[*] Computed adaptive K = {k} for N={N} with coeff={k_coeff}")
    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=False, logger=logger)
    snn = build_snn_graph(
        candidates, X=X, prune_method="inverse_k", 
        rescue_edges=True, rescue_cos_threshold=0.92, logger=logger
    )

    # Use profile discovery by default as requested
    ms_result = run_resolution_profile_discovery(
        snn,
        objective_function="CPM",
        per_component=True,
        min_component_size=50,
        logger=logger,
    )
    hierarchy = build_cluster_hierarchy(ms_result, logger=logger)

    stability = score_and_select_scales(
        hierarchy, ms_result, snn, 
        selection_policy="best_composite", consensus_runs=5, seed=0, logger=logger
    )
    
    logger.info(f"[*] Base Graph & Profile Pipeline Constructed in {time.time() - t_graph:.1f}s.")
    logger.info(f"[*] Handing over to Optuna for lightning-fast merge optimization (Trials={n_inner_trials})...")
    
    def objective(trial):
        centroid_cos = trial.suggest_float("centroid_cos", 0.85, 0.98, step=0.01)
        edge_conn = trial.suggest_float("edge_conn", 0.05, 0.30, step=0.01)
        homology_cos = trial.suggest_float("homology_cos", 0.90, 0.99, step=0.01)

        old_level = logger.level
        logger.setLevel(logging.CRITICAL) 
        
        try:
            refined = refine_and_flatten(
                X_pca, snn, hierarchy, stability, ms_result,
                centroid_cos_threshold=centroid_cos,
                edge_connectivity_threshold=edge_conn,
                output_level="fine",
                k_neighbors=k,
                homology_rescue=True,
                homology_rescue_cos=homology_cos,
                cross_branch_rescue=False,
                logger=logger,
            )
            
            metrics = evaluate_clustering(
                refined["labels_all"]["fine"], 
                prot_meta,
                graph=None, 
                level_name="fine", 
                extended_eval=False, 
                logger=logger
            )
            f1 = metrics["primary_score"]
        except Exception as e:
            logger.setLevel(old_level)
            logger.error(f"Error during merge test: {e}")
            raise optuna.TrialPruned()
        finally:
            logger.setLevel(old_level)
            
        print(f"    -> [k_coeff={k_coeff}] Trial #{trial.number:02d} | cos={centroid_cos:.2f} edge={edge_conn:.2f} hom={homology_cos:.2f}  =>  F1: {f1:.4f}")
        return f1

    study = optuna.create_study(direction="maximize", study_name=f"Evo_k{k_coeff}")
    study.enqueue_trial({"centroid_cos": 0.85, "edge_conn": 0.05, "homology_cos": 0.95})
    study.optimize(objective, n_trials=n_inner_trials)
    
    # Extract native data to easily pickle return across Joblib boundaries
    if len(study.trials) > 0:
        best_val = study.best_value
        best_params = study.best_trial.params
        logger.info(f"[+] BEST for k_coeff={k_coeff}: Trial #{study.best_trial.number} achieved F1 = {best_val:.4f}")
    else:
        best_val = -1
        best_params = {}
        
    return {
        "k_coeff": k_coeff, 
        "df": study.trials_dataframe(), 
        "best_value": best_val, 
        "best_params": best_params
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Bilevel Sweeper (Outer: joblib, Inner: optuna)")
    parser.add_argument("organism", type=str, help="Dataset organism code (e.g. pfal_pber)")
    parser.add_argument("--k_grid", type=str, default="0.3,0.45,0.6,0.75", help="Comma-separated list of k_coeffs to sweep")
    parser.add_argument("--inner_trials", type=int, default=30, help="Number of fast merge trials per k_coeff graph")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of outer loop graphs to construct in parallel")
    args = parser.parse_args()

    organism = args.organism
    k_coeffs = [float(k) for k in args.k_grid.split(",")]
    
    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)

    main_logger = setup_logging(f"{organism}_parallel_sweep_main")
    
    main_logger.info("=" * 60)
    main_logger.info(f"PARALLEL BILEVEL SWEEP: {organism}")
    main_logger.info(f"Target k_coeffs: {k_coeffs}")
    main_logger.info(f"Outer Parallel Workers (n_jobs): {args.n_jobs}")
    main_logger.info(f"Inner Range per Graph: {args.inner_trials} Trials")
    main_logger.info("=" * 60)

    # -------------------------------------------------------------
    # EXPENSIVE BASE-LEVEL SETUP (Embeddings load on main thread)
    # -------------------------------------------------------------
    t0_setup = time.time()
    main_logger.info("Loading Embeddings globally before dispatching workers...")
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=main_logger)
    X_raw = X_by_layer[34]
    prep = prepare_embeddings(X_raw, pca_dim=400, logger=main_logger)
    X, X_pca, N = prep["X"], prep["X_pca"], prep["N"]
    main_logger.info(f"Global Base Memory Initialized in {time.time() - t0_setup:.1f}s.")

    # -------------------------------------------------------------
    # PARALLEL OUTER LOOP EXECUTION
    # -------------------------------------------------------------
    main_logger.info(f"Dispatching outer loops to {args.n_jobs} parallel cores. Check sub-logs for specific progress!")
    
    all_studies_data = Parallel(n_jobs=args.n_jobs)(
        delayed(optimize_for_k)(k_coeff, organism, args.inner_trials, X, X_pca, N, prot_meta) 
        for k_coeff in k_coeffs
    )

    # -------------------------------------------------------------
    # AGGREGATE RESULTS
    # -------------------------------------------------------------
    best_global_f1 = -1
    best_global_config = None
    
    for data in all_studies_data:
        if data["best_value"] > best_global_f1:
            best_global_f1 = data["best_value"]
            best_global_config = {"k_coeff": data["k_coeff"], **data["best_params"]}

    print("\n" + "=" * 60)
    print("ALL SWEEPS CONCLUDED ACROSS CORES.")
    print(f"GLOBAL BEST F1: {best_global_f1:.4f}")
    if best_global_config:
        print("WINNING CONFIGURATION:")
        for k, v in best_global_config.items():
            print(f"  {k}: {v:.3f}")
    print("=" * 60)

    plot_nested_results(all_studies_data, organism)


if __name__ == "__main__":
    main()
