"""
run_simple_leiden.py — Simple kNN + Leiden (no SNN, no multiscale)

Runs Leiden clustering on a plain kNN graph with fixed k=40 and
resolution=0.6 by default, and reports pairwise plus orthogroup-type metrics.

Usage:
    cd Cluster_exps/src
    python run_simple_leiden.py pfal_pber
    python run_simple_leiden.py pfal_pber --k 40 --resolutions 0.6
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.io import setup_logging, load_embeddings, prepare_embeddings
from pipeline.evaluation import (
    extract_true_labels,
    extract_species_labels,
    compute_pairwise_confusion_metrics,
    compute_orthogroup_type_metrics,
)


def _metric_triplet(row, prefix):
    """Format compact P/R/F1 output for one orthogroup type."""
    return (
        f"{row.get(f'{prefix}_P', 0.0):.3f}/"
        f"{row.get(f'{prefix}_R', 0.0):.3f}/"
        f"{row.get(f'{prefix}_F1', 0.0):.3f}"
    )


def _flatten_orthogroup_type_metrics(orthogroup_types):
    """Flatten 1:1 / 1:m / n:m metrics into CSV-friendly columns."""
    flattened = {}
    column_prefixes = {
        "1:1": "oto",
        "1:m": "otm",
        "n:m": "ntm",
    }

    for metric_name, prefix in column_prefixes.items():
        values = orthogroup_types.get(metric_name, {})
        flattened[f"{prefix}_P"] = float(values.get("precision", 0.0))
        flattened[f"{prefix}_R"] = float(values.get("recall", 0.0))
        flattened[f"{prefix}_F1"] = float(values.get("f1", 0.0))
        flattened[f"{prefix}_n_proteins"] = int(values.get("n_proteins", 0))
        flattened[f"{prefix}_n_groups"] = int(values.get("n_groups", 0))

    return flattened


def main():
    parser = argparse.ArgumentParser(
        description="Simple kNN + Leiden clustering with fixed k/resolution defaults"
    )
    parser.add_argument("organism", type=str, help="Organism code (e.g. pfal_pber)")
    parser.add_argument("--k", type=int, default=40, help="Fixed kNN k (default: 40)")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="0.6",
        help="Comma-separated CPM resolutions (default: 0.6)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="CPM",
        choices=["CPM", "modularity"],
        help="Leiden objective function",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    organism = args.organism
    k = args.k
    resolutions = [float(r) for r in args.resolutions.split(",")]

    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)
    results_dir = f"../Model_Cluster_Results/simple_leiden/{species}"
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logging(f"{organism}_simple_leiden")
    logger.info("=" * 60)
    logger.info(f"Simple kNN + Leiden — {organism}")
    logger.info(f"  k = {k}  |  resolutions = {resolutions}")
    logger.info(f"  objective = {args.objective}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load + prepare embeddings
    # ------------------------------------------------------------------
    t0 = time.time()
    X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=logger)
    X_raw = X_by_layer[34]
    prep = prepare_embeddings(X_raw, pca_dim=400, logger=logger)
    X = prep["X"]  # L2-normalised, PCA-reduced, scaled
    N = prep["N"]
    true_labels = extract_true_labels(prot_meta)
    logger.info(f"  Loaded {N} proteins in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Build simple kNN graph (not SNN — just kNN with cosine weights)
    # ------------------------------------------------------------------
    logger.info(f"Building kNN graph with k={k} ...")
    from cluster_lib import _knn_edges

    edges, weights = _knn_edges(
        X,
        n_neighbors=k,
        metric="cosine",
        use_faiss=True,
        mutual_knn=False,  # plain kNN, not mutual
    )

    import igraph as ig

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    logger.info(
        f"  kNN graph: {g.vcount()} nodes, {g.ecount()} edges, "
        f"mean degree {2*g.ecount()/N:.1f}"
    )

    # ------------------------------------------------------------------
    # 3. Leiden at each resolution
    # ------------------------------------------------------------------
    import leidenalg as la

    if args.objective.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition

    records = []
    for res in resolutions:
        t_r = time.time()
        part = la.find_partition(
            g,
            partition_type,
            weights=g.es["weight"],
            resolution_parameter=res,
            n_iterations=-1,
            seed=args.seed,
        )
        labels = np.array(part.membership, dtype=np.int32)
        n_clusters = len(set(labels.tolist()))
        n_singletons = sum(1 for cid in set(labels) if np.sum(labels == cid) == 1)

        pairwise = compute_pairwise_confusion_metrics(true_labels, labels)
        species_labels = extract_species_labels(prot_meta)
        orthogroup_types = compute_orthogroup_type_metrics(
            true_labels,
            labels,
            species_labels,
            logger=logger,
        )
        elapsed = time.time() - t_r

        record = {
            "resolution": res,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "precision": pairwise["precision"],
            "recall": pairwise["recall"],
            "f1": pairwise["f1"],
            "TP": pairwise["TP"],
            "FP": pairwise["FP"],
            "FN": pairwise["FN"],
            "time_s": elapsed,
        }
        record.update(_flatten_orthogroup_type_metrics(orthogroup_types))
        records.append(record)

        logger.info(
            f"  res={res:.2f}  K={n_clusters:>5d}  "
            f"P={pairwise['precision']:.4f}  R={pairwise['recall']:.4f}  "
            f"F1={pairwise['f1']:.4f}  "
            f"1:1(P/R/F1)={_metric_triplet(record, 'oto')}  "
            f"n:m(P/R/F1)={_metric_triplet(record, 'ntm')}  "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # 4. Save CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, f"{organism}_simple_leiden_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved results CSV to: {csv_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"  Simple Leiden Results — {organism} (k={k})")
    print("=" * 115)
    print(
        f"  {'Res':>6s}  {'K':>6s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  "
        f"{'1:1 P/R/F1':>17s}  {'n:m P/R/F1':>17s}"
    )
    print("  " + "-" * 110)
    for _, row in df.iterrows():
        print(
            f"  {row['resolution']:6.2f}  {int(row['n_clusters']):6d}  "
            f"{row['precision']:10.4f}  {row['recall']:10.4f}  {row['f1']:10.4f}  "
            f"{_metric_triplet(row, 'oto'):>17s}  {_metric_triplet(row, 'ntm'):>17s}"
        )
    print("=" * 115)

    # ------------------------------------------------------------------
    # 6. Plot Precision & Recall vs Resolution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        df["resolution"], df["precision"],
        marker="o", linewidth=2.2, markersize=8,
        color="#4e79a7", label="Precision",
    )
    ax.plot(
        df["resolution"], df["recall"],
        marker="s", linewidth=2.2, markersize=8,
        color="#e15759", label="Recall",
    )
    ax.plot(
        df["resolution"], df["f1"],
        marker="D", linewidth=2, markersize=7,
        color="#59a14f", linestyle="--", alpha=0.8, label="F1",
    )

    # Annotate data points
    for _, row in df.iterrows():
        ax.annotate(
            f"K={int(row['n_clusters'])}",
            (row["resolution"], row["f1"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
            color="#444444",
        )

    ax.set_xlabel("Leiden CPM Resolution", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Simple Leiden (k={k}) — Precision / Recall vs Resolution\n{organism}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(min(resolutions) - 0.05, max(resolutions) + 0.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()

    img_path = os.path.join(results_dir, f"{organism}_simple_leiden_pr_tradeoff.png")
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved P/R plot to: {img_path}")
    print(f"\nPlot saved to: {img_path}")


if __name__ == "__main__":
    main()
