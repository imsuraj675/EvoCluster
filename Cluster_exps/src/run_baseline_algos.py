"""
run_baseline_algos.py — Run baseline clustering algorithms for the results table.

Evaluates HDBSCAN and Agglomerative clustering on pfal_pber
(or any organism) using the same embeddings + PCA(400) pipeline,
reporting pairwise P/R/F1 for each method + parameter configuration.

Usage:
    cd Cluster_exps/src
    python run_baseline_algos.py pfal_pber
    python run_baseline_algos.py mmus_hsap
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from pipeline.io import setup_logging, load_embeddings, prepare_embeddings
from pipeline.evaluation import (
    extract_true_labels,
    compute_pairwise_confusion_metrics,
)
from sklearn.preprocessing import StandardScaler


def evaluate_labels(true_labels, pred_labels):
    """Compute pairwise P/R/F1 + cluster stats."""
    pairwise = compute_pairwise_confusion_metrics(true_labels, pred_labels)
    pred_np = np.asarray(pred_labels)
    n_clusters = len(set(int(l) for l in pred_np if l >= 0))
    n_singletons = int(np.sum(pred_np == -1))
    return {
        **pairwise,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
    }




# =====================================================================
# HDBSCAN
# =====================================================================
def run_hdbscan(X, min_cluster_size, min_samples, allow_single_cluster, logger):
    logger.info(
        f"  HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
        f"allow_single_cluster={allow_single_cluster}"
    )
    t0 = time.time()

    # Try sklearn.cluster.HDBSCAN first (sklearn >= 1.3), then fall back
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        model = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            allow_single_cluster=allow_single_cluster,
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
    except ImportError:
        try:
            import hdbscan

            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                allow_single_cluster=allow_single_cluster,
            )
            labels = model.fit_predict(X)
        except ImportError:
            raise ImportError(
                "HDBSCAN not available. Either upgrade scikit-learn>=1.3 "
                "or pip install hdbscan"
            )

    elapsed = time.time() - t0
    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels[labels >= 0].tolist()))
    logger.info(
        f"    Done in {elapsed:.1f}s — {n_clusters} clusters, {n_noise} noise points"
    )
    return labels, elapsed


# =====================================================================
# Agglomerative (with kNN connectivity)
# =====================================================================
def run_agglomerative(X, distance_threshold, linkage, n_neighbors, logger):
    from sklearn.cluster import AgglomerativeClustering
    from cluster_lib import build_knn_connectivity

    logger.info(
        f"  Agglomerative: dist_thresh={distance_threshold}, "
        f"linkage={linkage}, n_neighbors={n_neighbors}"
    )
    t0 = time.time()

    connectivity = build_knn_connectivity(
        X, n_neighbors=n_neighbors, metric="cosine", use_faiss=True
    )

    # sklearn >= 1.2 uses 'metric', older versions use 'affinity'
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
            connectivity=connectivity,
            distance_threshold=distance_threshold,
            compute_full_tree=True,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage=linkage,
            connectivity=connectivity,
            distance_threshold=distance_threshold,
            compute_full_tree=True,
        )

    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    n_clusters = len(set(labels[labels >= 0].tolist()))
    logger.info(f"    Done in {elapsed:.1f}s — {n_clusters} clusters")
    return labels, elapsed


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run baseline clustering algorithms for the results table"
    )
    parser.add_argument("organism", type=str, help="Organism code (e.g. pfal_pber)")
    args = parser.parse_args()

    organism = args.organism
    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)
    results_dir = f"../Model_Cluster_Results/baselines/{species}"
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logging(f"{organism}_baselines")

    logger.info("=" * 60)
    logger.info(f"BASELINE ALGORITHMS — {organism}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load + prepare embeddings (same pipeline as multiscale)
    # ------------------------------------------------------------------
    t0 = time.time()
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=logger)
    X_raw = X_by_layer[34]
    prep = prepare_embeddings(X_raw, pca_dim=400, logger=logger)
    X_normed = prep["X"]  # L2-normalised
    X_pca = prep["X_pca"]  # PCA-reduced, not normalised
    N = prep["N"]
    true_labels = extract_true_labels(prot_meta)
    n_true_groups = len(set(true_labels.tolist()))
    logger.info(f"  {N} proteins, {n_true_groups} true orthogroups")
    logger.info(f"  Embedding load+prep took {time.time()-t0:.1f}s")

    # Standard-scale the PCA embeddings for algorithms that assume Gaussian
    X_scaled = StandardScaler().fit_transform(X_pca)

    # ------------------------------------------------------------------
    # 2. Define parameter grids for each method
    # ------------------------------------------------------------------
    all_records = []


    # ── HDBSCAN ──
    # Note: HDBSCAN requires min_cluster_size >= 2 internally.
    # Noise points (label=-1) are treated as singletons in our evaluator.
    hdbscan_configs = [
        {"min_cluster_size": 2, "min_samples": 1},
        {"min_cluster_size": 2, "min_samples": 3},
        {"min_cluster_size": 3, "min_samples": 2},
        {"min_cluster_size": 5, "min_samples": 3},
        {"min_cluster_size": 5, "min_samples": 5},
        {"min_cluster_size": 10, "min_samples": 5},
        {"min_cluster_size": 15, "min_samples": 10},
        {"min_cluster_size": 20, "min_samples": 10},
    ]

    logger.info(f"\n{'─'*60}")
    logger.info(f"HDBSCAN  ({len(hdbscan_configs)} configs)")
    logger.info(f"{'─'*60}")

    for cfg in hdbscan_configs:
        try:
            labels, elapsed = run_hdbscan(
                X_scaled, cfg["min_cluster_size"], cfg["min_samples"],
                allow_single_cluster=True, logger=logger,
            )
            metrics = evaluate_labels(true_labels, labels)
            record = {
                "method": "HDBSCAN",
                "params": f"mcs={cfg['min_cluster_size']},ms={cfg['min_samples']}",
                **metrics,
                "time_s": elapsed,
            }
            all_records.append(record)
            logger.info(
                f"    mcs={cfg['min_cluster_size']:>3d} ms={cfg['min_samples']:>3d}  "
                f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}  K={metrics['n_clusters']}  "
                f"noise={metrics['n_singletons']}"
            )
        except Exception as e:
            logger.error(f"    HDBSCAN failed with {cfg}: {e}")

    # ── Agglomerative ──
    agglo_configs = [
        {"distance_threshold": 0.10, "linkage": "average", "n_neighbors": 40},
        {"distance_threshold": 0.15, "linkage": "average", "n_neighbors": 40},
        {"distance_threshold": 0.20, "linkage": "average", "n_neighbors": 40},
        {"distance_threshold": 0.25, "linkage": "average", "n_neighbors": 40},
        {"distance_threshold": 0.30, "linkage": "average", "n_neighbors": 40},
    ]

    logger.info(f"\n{'─'*60}")
    logger.info(f"AGGLOMERATIVE  ({len(agglo_configs)} configs)")
    logger.info(f"{'─'*60}")

    for cfg in agglo_configs:
        try:
            labels, elapsed = run_agglomerative(
                X_normed,  # use L2-normalised for cosine metric
                cfg["distance_threshold"],
                cfg["linkage"],
                cfg["n_neighbors"],
                logger,
            )
            metrics = evaluate_labels(true_labels, labels)
            record = {
                "method": "Agglomerative",
                "params": (
                    f"dt={cfg['distance_threshold']:.2f},"
                    f"link={cfg['linkage']},nn={cfg['n_neighbors']}"
                ),
                **metrics,
                "time_s": elapsed,
            }
            all_records.append(record)
            logger.info(
                f"    dt={cfg['distance_threshold']:.2f} link={cfg['linkage']}  "
                f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}  K={metrics['n_clusters']}"
            )
        except Exception as e:
            logger.error(f"    Agglomerative failed with {cfg}: {e}")

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(results_dir, f"{organism}_baseline_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved results to: {csv_path}")

    # ------------------------------------------------------------------
    # 4. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"  Baseline Clustering Results — {organism}")
    print("=" * 90)
    print(
        f"  {'Method':<15s}  {'Params':<30s}  "
        f"{'K':>6s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Time':>6s}"
    )
    print("  " + "-" * 85)
    for _, row in df.iterrows():
        print(
            f"  {row['method']:<15s}  {row['params']:<30s}  "
            f"{int(row['n_clusters']):6d}  {row['precision']:10.4f}  "
            f"{row['recall']:10.4f}  {row['f1']:10.4f}  {row['time_s']:5.1f}s"
        )
    print("=" * 90)

    # Find best per method
    print("\nBest per method:")
    for method in ["HDBSCAN", "Agglomerative"]:
        sub = df[df["method"] == method]
        if len(sub) > 0:
            best = sub.loc[sub["f1"].idxmax()]
            print(
                f"  {method:<15s}  {best['params']:<30s}  "
                f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}"
            )

    # ------------------------------------------------------------------
    # 5. Bar chart of best F1 per method
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = []
    best_f1s = []
    best_ps = []
    best_rs = []

    for method in ["HDBSCAN", "Agglomerative"]:
        sub = df[df["method"] == method]
        if len(sub) > 0:
            best = sub.loc[sub["f1"].idxmax()]
            methods.append(f"{method}\n({best['params'][:25]})")
            best_f1s.append(best["f1"])
            best_ps.append(best["precision"])
            best_rs.append(best["recall"])

    x_pos = np.arange(len(methods))
    bar_width = 0.25

    bars_p = ax.bar(x_pos - bar_width, best_ps, bar_width, label="Precision", color="#4e79a7")
    bars_r = ax.bar(x_pos, best_rs, bar_width, label="Recall", color="#e15759")
    bars_f = ax.bar(x_pos + bar_width, best_f1s, bar_width, label="F1", color="#59a14f")

    # Value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Baseline Clustering Comparison — {organism}\n(Best config per method)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    img_path = os.path.join(results_dir, f"{organism}_baseline_comparison.png")
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison chart to: {img_path}")
    print(f"\nChart saved to: {img_path}")


if __name__ == "__main__":
    main()
