import os
import time
import argparse
import numpy as np
import logging
import optuna
import pandas as pd
import matplotlib.pyplot as plt

# Import the core pipeline components directly
from pipeline.io import setup_logging, log_device_info, load_embeddings, prepare_embeddings
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph
from pipeline.leiden import run_leiden_multiscale, build_cluster_hierarchy
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import get_ground_truth_stats, evaluate_clustering

def plot_results(study, organism):
    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    
    if len(df) == 0:
        return
        
    params = [c for c in df.columns if c.startswith('params_')]
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 5))
    if len(params) == 1:
        axes = [axes]
        
    for ax, p in zip(axes, params):
        ax.scatter(df[p], df['value'], alpha=0.7, color='teal', edgecolor='black')
        ax.set_xlabel(p.replace('params_', ''))
        ax.set_ylabel('Pairwise F1')
        ax.set_title(f'F1 vs {p.replace("params_", "")}')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    out_img = f"sweep_fast_{organism}_params.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    
    out_csv = f"sweep_fast_{organism}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved rapid visualization to: {out_img}")
    print(f"Saved raw tracking data to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Lightning Fast In-Memory Sweeper for Merge Parameters")
    parser.add_argument("organism", type=str, help="Dataset organism code (e.g. pfal_pber)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of merge trials to run")
    parser.add_argument("--k_coeff", type=float, default=0.6, help="Fixed k_coeff for the graph base")
    args = parser.parse_args()

    organism = args.organism
    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)

    logger = setup_logging(f"{organism}_fast_sweep")
    
    logger.info("=" * 60)
    logger.info(f"FAST IN-MEMORY SWEEP: {organism}")
    logger.info("This script runs Embeddings -> Graph -> Leiden -> Selection exactly ONCE.")
    logger.info("Then it loops *only* the fast Merge step over the Optuna Grid.")
    logger.info("=" * 60)

    # -------------------------------------------------------------
    # EXPENSIVE ONE-TIME SETUP (Steps 1 through 4)
    # -------------------------------------------------------------
    t0_setup = time.time()
    
    # 1. Load Embeddings
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=logger)
    X_raw = X_by_layer[34]
    prep = prepare_embeddings(X_raw, pca_dim=400, logger=logger)
    X, X_pca, N = prep["X"], prep["X_pca"], prep["N"]

    # 2. SNN Graph
    k = compute_adaptive_k(N, coeff=args.k_coeff, cap=150)
    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=False, logger=logger)
    snn = build_snn_graph(
        candidates, X=X, prune_method="inverse_k", 
        rescue_edges=True, rescue_cos_threshold=0.92, logger=logger
    )

    # 3. Leiden Sweeps (using manual mode for speed, avoiding PyGen)
    ms_result = run_leiden_multiscale(
        snn, resolution_grid=None, objective_function="CPM", seed=0, logger=logger
    )
    hierarchy = build_cluster_hierarchy(ms_result, logger=logger)

    # 4. Score and Select Scales
    stability = score_and_select_scales(
        hierarchy, ms_result, snn,
        selection_policy="best_composite", consensus_runs=5, seed=0, logger=logger
    )
    
    logger.info(f"\n[+] ONE-TIME EXPENSIVE SETUP COMPLETED IN {time.time() - t0_setup:.1f} SECONDS.")
    logger.info("[+] COMMENCING HIGH-SPEED OPTUNA MERGE SWEEP...\n")

    # -------------------------------------------------------------
    # LIGHTNING FAST OPTUNA LOOP (Steps 5 and 6)
    # -------------------------------------------------------------
    def objective(trial):
        # We sweep only the merge parameters here
        centroid_cos = trial.suggest_float("centroid_cos", 0.85, 0.98, step=0.01)
        edge_conn = trial.suggest_float("edge_conn", 0.05, 0.30, step=0.01)
        homology_cos = trial.suggest_float("homology_cos", 0.90, 0.99, step=0.01)

        t_trial = time.time()
        
        # Disable logging for the loops to avoid massive log file spam
        old_level = logger.level
        logger.setLevel(logging.CRITICAL) 
        
        try:
            # Step 5: Refine (Cascade Merge)
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

            # Step 6: Evaluate (Only Fine)
            metrics = evaluate_clustering(
                refined["labels_all"]["fine"],
                prot_meta,
                graph=None, 
                level_name="fine",
                extended_eval=False,
                logger=logger,
            )
            
            f1 = metrics["primary_score"] # This is Pairwise F1
        finally:
            # Restore logging
            logger.setLevel(old_level)
            
        print(f"  Trial #{trial.number:02d} | cos:{centroid_cos:.2f} edge:{edge_conn:.2f} hom:{homology_cos:.2f}  => F1: {f1:.4f}  (Took {time.time() - t_trial:.1f}s)")
        return f1

    # Execute study inside this single python process
    study = optuna.create_study(direction="maximize", study_name=f"EvoFastSweep_{organism}")
    study.enqueue_trial({"centroid_cos": 0.85, "edge_conn": 0.05, "homology_cos": 0.95})
    
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n[!] User interrupted Fast Sweep.")
        
    print("\n" + "=" * 60)
    if len(study.trials) > 0 and study.best_trial:
        print(f"BEST RUN: Trial #{study.best_trial.number} (Pairwise F1 = {study.best_value:.4f})")
        print("BEST PARAMETERS:")
        for k_param, v in study.best_trial.params.items():
            print(f"  {k_param}: {v:.2f}")
    print("=" * 60)

    plot_results(study, organism)

if __name__ == "__main__":
    main()
