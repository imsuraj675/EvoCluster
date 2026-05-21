"""
compare_knn_snn.py — Peak F1 Comparison: KNN vs SNN Graph

Builds both a raw KNN graph and a Jaccard-pruned SNN graph.
Because the edge weight scales differ drastically (cosine ~0.9 vs Jaccard ~0.2), 
this script sweeps 30 different Leiden CPM resolutions (0.01 to 0.99) for BOTH 
graphs and plots the MAXIMUM achievable F1 for each architecture.

Usage:
    cd Cluster_exps/src
    python compare_knn_snn.py pfal_pber drer_xtro mmus_hsap cbov_cpar_ncan_tgon_tbrt_pber_pfal
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la
from scipy import sparse
from joblib import Parallel, delayed

from pipeline.io import setup_logging, load_embeddings, prepare_embeddings
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph, build_knn_graph
from pipeline.evaluation import extract_true_labels, compute_pairwise_confusion_metrics

def run_single_leiden(graph_result, resolution, seed=0):
    g = graph_result["graph"]
    weights = g.es["weight"] if "weight" in g.edge_attributes() else None
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    return np.array(part.membership, dtype=np.int32)

def sweep_best_leiden(graph_result, true_labels, resolutions):
    """Scan resolutions and return metrics for the peak F1 configuration."""
    best_f1 = -1.0
    best_metrics = None
    best_res = -1.0
    best_k = 0
    
    for res in resolutions:
        labels = run_single_leiden(graph_result, res)
        metrics = compute_pairwise_confusion_metrics(true_labels, labels)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics
            best_res = res
            best_k = len(set(labels[labels >= 0].tolist()))
            
    return best_metrics, best_res, best_k

def plot_grid_comparison(all_metrics, save_path):
    """Create a grid of grouped bar charts comparing Peak KNN vs Peak SNN."""
    metrics_names = ["Precision", "Recall", "F1"]
    
    n_datasets = len(all_metrics)
    cols = 2 if n_datasets > 1 else 1
    rows = math.ceil(n_datasets / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    width = 0.32
    x = np.arange(len(metrics_names))
    
    for idx, (organism, data) in enumerate(all_metrics.items()):
        ax = axes[idx]
        knn_vals = [data['knn']["precision"], data['knn']["recall"], data['knn']["f1"]]
        snn_vals = [data['snn']["precision"], data['snn']["recall"], data['snn']["f1"]]

        bars_knn = ax.bar(x - width / 2, knn_vals, width, label="Peak KNN",
                           color="#e15759", edgecolor="black", linewidth=0.8, zorder=3)
        bars_snn = ax.bar(x + width / 2, snn_vals, width, label="Peak SNN (Jaccard)",
                           color="#4e79a7", edgecolor="black", linewidth=0.8, zorder=3)

        # Annotate bars
        for bar in bars_knn:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for bar in bars_snn:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Add resolution context text box
        res_text = (f"KNN best res={data['knn_res']:.2f}\n"
                    f"SNN best res={data['snn_res']:.2f}")
        ax.text(0.02, 0.95, res_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f"{organism}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=12)
        ax.set_ylim(0, max(max(knn_vals), max(snn_vals)) * 1.25)
        if idx == 0:
            ax.legend(fontsize=11, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("PEAK Capability: KNN vs SNN (Optimal Resolution Sweeps)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved Peak comparison plot to: {save_path}")


def process_dataset(organism, k_coeff):
    dataset_path = "../OrthoMCL/"
    species = f"{organism}_filtered"
    fasta_path = os.path.join(dataset_path, f"{organism}.fasta")
    emb_path = os.path.join(dataset_path, species)

    logger = setup_logging(f"{organism}_knn_vs_snn_peak")
    logger.info(f"Processing {organism}...")

    try:
        X_by_layer, prot_meta = load_embeddings(fasta_path, emb_path, [34], logger=logger)
        X_raw = X_by_layer[34]
        prep = prepare_embeddings(X_raw, pca_dim=400, logger=logger)
        X = prep["X"]
        N = prep["N"]
        true_labels = extract_true_labels(prot_meta)
        
        k = compute_adaptive_k(N, coeff=k_coeff, cap=150)
        candidates = build_candidate_neighbors(X, k_candidates=k, use_gpu=False, logger=logger)
        
        # Build both architectures
        knn_result = build_knn_graph(candidates, logger=logger)
        snn_result = build_snn_graph(candidates, X=X, prune_method="inverse_k", rescue_edges=False, logger=logger)

        # 30-step resolution sweep
        resolutions = np.linspace(0.01, 0.99, 30)
        
        logger.info(f"Sweeping 30 resolutions on KNN...")
        knn_metrics, knn_res, knn_k = sweep_best_leiden(knn_result, true_labels, resolutions)
        
        logger.info(f"Sweeping 30 resolutions on SNN...")
        snn_metrics, snn_res, snn_k = sweep_best_leiden(snn_result, true_labels, resolutions)

        return organism, knn_metrics, snn_metrics, knn_res, snn_res, knn_k, snn_k, None
    except Exception as e:
        logger.error(f"Failed processing {organism}: {e}")
        return organism, None, None, None, None, None, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Peak KNN vs SNN Graph Comparison (Swept Resolutions)")
    parser.add_argument("datasets", nargs="+", help="List of dataset organism codes")
    parser.add_argument("--k_coeff", type=float, default=0.6, help="k = coeff * sqrt(N) (default: 0.6)")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"PEAK KNN vs SNN COMPARISON: {len(args.datasets)} datasets")
    print(f"Sweeping 30 CPM resolutions (0.01 -> 0.99) to find max F1.")
    print("=" * 60)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_dataset)(org, args.k_coeff) for org in args.datasets
    )

    all_metrics = {}
    print("\n" + "=" * 60)
    print("RESULTS (PEAK F1)")
    print("=" * 60)
    for org, knn_m, snn_m, knn_res, snn_res, k_knn, k_snn, error in results:
        if error:
            print(f"  [!] {org}: FAILED - {error}")
        else:
            all_metrics[org] = {
                'knn': knn_m, 'snn': snn_m, 
                'knn_res': knn_res, 'snn_res': snn_res
            }
            print(f"  {org}:")
            print(f"    Peak KNN (res={knn_res:.2f}): K={k_knn:<5d}  P={knn_m['precision']:.4f}  R={knn_m['recall']:.4f}  F1={knn_m['f1']:.4f}")
            print(f"    Peak SNN (res={snn_res:.2f}): K={k_snn:<5d}  P={snn_m['precision']:.4f}  R={snn_m['recall']:.4f}  F1={snn_m['f1']:.4f}")
    print("=" * 60)

    if all_metrics:
        plot_grid_comparison(all_metrics, "knn_vs_snn_peak_all.png")

if __name__ == "__main__":
    main()
