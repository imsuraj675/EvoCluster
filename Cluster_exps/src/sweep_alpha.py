"""
sweep_alpha.py — Sweep SGC diffusion mixing coefficient (alpha).

Builds graph + hierarchy once, precomputes X_diff once, then for each alpha
value computes X_alpha and runs only the merge + evaluate steps.

Usage:
    cd Cluster_exps/src
    python sweep_alpha.py pfal_pber
    python sweep_alpha.py pfal_pber --alphas 0.0,0.2,0.4,0.6,0.8,0.9,1.0
"""

import os
import time
import argparse
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.io import setup_logging, load_embeddings, prepare_embeddings
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph
from pipeline.leiden import build_cluster_hierarchy
from pipeline.discovery import run_resolution_profile_discovery
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation import evaluate_clustering
from pipeline.diffusion import compute_diffused_embeddings


def main():
    parser = argparse.ArgumentParser(description="Alpha Sweep for SGC Feature Diffusion")
    parser.add_argument("organism", type=str, help="Dataset organism code (e.g. pfal_pber)")
    parser.add_argument("--k_coeff", type=float, default=0.6, help="Fixed k_coeff (default: 0.6)")
    parser.add_argument(
        "--alphas", type=str, default="0.0,0.2,0.4,0.6,0.8,0.9,1.0",
        help="Comma-separated alpha values to sweep",
    )
    args = parser.parse_args()

    organism = args.organism
    alphas = [float(a) for a in args.alphas.split(",")]

    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)

    logger = setup_logging(f"{organism}_alpha_sweep")

    logger.info("=" * 60)
    logger.info(f"ALPHA SWEEP: {organism}")
    logger.info(f"k_coeff = {args.k_coeff}")
    logger.info(f"Alpha values: {alphas}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # ONE-TIME SETUP (Steps 1 through 4)
    # -----------------------------------------------------------------
    t0_setup = time.time()

    # 1. Load Embeddings
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=logger)
    X_raw = X_by_layer[34]
    prep = prepare_embeddings(X_raw, pca_dim=400, logger=logger)
    X, X_pca, N = prep["X"], prep["X_pca"], prep["N"]

    # 2. SNN Graph
    k = compute_adaptive_k(N, coeff=args.k_coeff, cap=250)
    candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=False, logger=logger)
    snn = build_snn_graph(
        candidates, X=X, prune_method="inverse_k",
        rescue_edges=True, rescue_cos_threshold=0.92, logger=logger
    )

    # 3. Discovery + Hierarchy
    ms_result = run_resolution_profile_discovery(
        snn,
        objective_function="CPM",
        per_component=True,
        min_component_size=50,
        logger=logger,
    )
    hierarchy = build_cluster_hierarchy(ms_result, logger=logger)

    # 4. Score and Select Scales
    stability = score_and_select_scales(
        hierarchy, ms_result, snn,
        selection_policy="best_composite", consensus_runs=5, seed=0, logger=logger
    )

    # 5. Precompute X_diff ONCE
    _, X_diff = compute_diffused_embeddings(X_pca, snn["adjacency"], alpha=1.0, logger=logger)

    logger.info(f"\n[+] ONE-TIME SETUP COMPLETED IN {time.time() - t0_setup:.1f}s")
    logger.info(f"[+] COMMENCING ALPHA SWEEP ({len(alphas)} values)...\n")

    # -----------------------------------------------------------------
    # ALPHA SWEEP LOOP
    # -----------------------------------------------------------------
    records = []

    for alpha in alphas:
        t_trial = time.time()
        logger.info(f"\n--- Alpha = {alpha:.2f} ---")

        # Compute X_alpha on the fly
        if alpha > 0:
            X_alpha = ((1.0 - alpha) * X_pca + alpha * X_diff).astype(np.float32)
        else:
            X_alpha = None

        # Suppress merge logging
        old_level = logger.level
        logger.setLevel(logging.CRITICAL)

        try:
            refined = refine_and_flatten(
                X_pca, snn, hierarchy, stability, ms_result,
                centroid_cos_threshold=0.85,
                edge_connectivity_threshold=0.05,
                output_level="fine",
                k_neighbors=k,
                homology_rescue=True,
                homology_rescue_cos=0.95,
                cross_branch_rescue=False,
                diffusion_alpha=alpha,
                diffusion_X_alpha=X_alpha,
                logger=logger,
            )

            metrics = evaluate_clustering(
                refined["labels_all"]["fine"],
                prot_meta,
                graph=None,
                level_name="fine",
                extended_eval=False,
                logger=logger,
            )
        finally:
            logger.setLevel(old_level)

        pairwise = metrics["pairwise"]
        og_types = metrics.get("orthogroup_types", {})

        record = {
            "alpha": alpha,
            "precision": pairwise["precision"],
            "recall": pairwise["recall"],
            "f1": pairwise["f1"],
            "n_clusters": metrics["n_clusters"],
            "n_merges": refined["n_merges"],
            "oto_P": og_types.get("1:1", {}).get("precision", 0.0),
            "oto_R": og_types.get("1:1", {}).get("recall", 0.0),
            "oto_F1": og_types.get("1:1", {}).get("f1", 0.0),
            "otm_P": og_types.get("1:m", {}).get("precision", 0.0),
            "otm_R": og_types.get("1:m", {}).get("recall", 0.0),
            "otm_F1": og_types.get("1:m", {}).get("f1", 0.0),
            "ntm_P": og_types.get("n:m", {}).get("precision", 0.0),
            "ntm_R": og_types.get("n:m", {}).get("recall", 0.0),
            "ntm_F1": og_types.get("n:m", {}).get("f1", 0.0),
            "time_s": time.time() - t_trial,
        }
        records.append(record)

        print(
            f"  alpha={alpha:.2f}  |  "
            f"P={pairwise['precision']:.4f}  R={pairwise['recall']:.4f}  F1={pairwise['f1']:.4f}  |  "
            f"K={metrics['n_clusters']}  merges={refined['n_merges']}  |  "
            f"1:1 F1={record['oto_F1']:.3f}  1:m F1={record['otm_F1']:.3f}  n:m F1={record['ntm_F1']:.3f}  |  "
            f"{record['time_s']:.1f}s"
        )

    # -----------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------
    df = pd.DataFrame(records)

    # Best alpha by F1
    best_row = df.loc[df["f1"].idxmax()]
    print("\n" + "=" * 60)
    print(f"BEST ALPHA: {best_row['alpha']:.2f}")
    print(f"  F1:        {best_row['f1']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall:    {best_row['recall']:.4f}")
    print(f"  1:1 F1:    {best_row['oto_F1']:.4f}")
    print(f"  1:m F1:    {best_row['otm_F1']:.4f}")
    print(f"  n:m F1:    {best_row['ntm_F1']:.4f}")
    print("=" * 60)

    # Save CSV
    csv_path = f"sweep_alpha_{organism}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------
    # 1. F1 vs alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["alpha"], df["f1"], marker="o", linewidth=2, color="indigo", label="Global F1")
    ax.plot(df["alpha"], df["oto_F1"], marker="s", linewidth=1.5, color="#59a14f", alpha=0.7, label="1:1 F1")
    ax.plot(df["alpha"], df["otm_F1"], marker="^", linewidth=1.5, color="#f28e2b", alpha=0.7, label="1:m F1")
    ax.plot(df["alpha"], df["ntm_F1"], marker="D", linewidth=1.5, color="#e15759", alpha=0.7, label="n:m F1")
    ax.set_xlabel("Diffusion Alpha")
    ax.set_ylabel("Pairwise F1")
    ax.set_title(f"F1 vs Diffusion Alpha — {organism}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    f1_img = f"sweep_alpha_{organism}_f1.png"
    fig.savefig(f1_img, dpi=150)
    plt.close(fig)
    print(f"Saved F1 vs alpha plot to: {f1_img}")

    # 2. P vs R scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        df["recall"], df["precision"],
        c=df["alpha"], cmap="coolwarm", s=80, edgecolor="black", linewidth=0.5,
    )
    for _, row in df.iterrows():
        ax.annotate(f"α={row['alpha']:.1f}", (row["recall"], row["precision"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.colorbar(sc, label="Alpha")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision vs Recall — {organism}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    pr_img = f"sweep_alpha_{organism}_pr.png"
    fig.savefig(pr_img, dpi=150)
    plt.close(fig)
    print(f"Saved P-R scatter to: {pr_img}")


if __name__ == "__main__":
    main()
