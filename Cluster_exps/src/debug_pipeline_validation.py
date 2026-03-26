import numpy as np
import time
import logging
from collections import Counter
from sklearn.preprocessing import normalize

# Import pipeline modules
from pipeline.graph import compute_adaptive_k, build_candidate_neighbors, build_snn_graph
from pipeline.discovery import run_resolution_profile_discovery, run_pygenstability_discovery
from pipeline.leiden import build_cluster_hierarchy
from pipeline.selection import score_and_select_scales
from pipeline.merge import refine_and_flatten
from pipeline.evaluation_cdlib import evaluate_topology_cdlib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
log = logging.getLogger("multiscale")
log.setLevel(logging.INFO)

def main():
    print("\n===============================")
    print("  SYNTHETIC PIPELINE VALIDATION")
    print("===============================\n")

    # Step 1 - Generate Synthetic Data
    from sklearn.datasets import make_blobs
    N, D = 500, 256
    X_raw, _ = make_blobs(n_samples=N, n_features=D, centers=8, random_state=42)
    X = normalize(X_raw, norm='l2', axis=1)
    
    log.info(f"Generated synthetic embeddings: N={N}, D={D}")

    # Step 2 - Graph Construction
    k = compute_adaptive_k(N)
    candidates = build_candidate_neighbors(X, k_candidates=k, use_faiss=False, logger=log)
    snn_result = build_snn_graph(candidates, prune_method="inverse_k", logger=log)

    # Step 3 - Scale Discovery (Try PyGen, fallback to Resolution Profile)
    import traceback
    try:
        from pygenstability import run
        ms_result = run_pygenstability_discovery(snn_result, logger=log)
        if len(ms_result["levels"]) < 3:
            log.warning("PyGen returned < 3 feasible scales. Falling back to resolution profile.")
            ms_result = run_resolution_profile_discovery(snn_result, resolution_range=(0.01, 1.0), logger=log)
    except ImportError:
        log.warning("PyGenStability not installed. Using resolution profile.")
        ms_result = run_resolution_profile_discovery(snn_result, resolution_range=(0.01, 1.0), logger=log)
    except Exception as e:
        log.error("PyGen stability crashed! Stack trace below:")
        log.error(traceback.format_exc())
        ms_result = run_resolution_profile_discovery(snn_result, resolution_range=(0.01, 1.0), logger=log)
        
    hierarchy = build_cluster_hierarchy(ms_result, logger=log)

    # Step 4 - Scale Selection (Task C fixes)
    stability = score_and_select_scales(hierarchy, ms_result, snn_result, logger=log)

    # Step 5 - Merge Stage (Task E fixes)
    # Output level "fine" (Task D fix)
    merged = refine_and_flatten(
        X, snn_result, hierarchy, stability, ms_result, 
        output_level="fine", k_neighbors=k, logger=log
    )
    fine_labels = merged["labels"]

    # Step 6 - CDlib Evaluation (Phase 4E fix)
    cdlib_metrics = evaluate_topology_cdlib(snn_result["graph"], fine_labels, logger=log)

    # Step 7 - Print Validation Summary
    print("\n===============================")
    print("      VALIDATION SUMMARY       ")
    print("===============================\n")

    print("[GRAPH]")
    stats = snn_result['stats']
    print(f"  Nodes: {snn_result['n_nodes']}, Edges: {snn_result['n_edges']}, Mean Degree: {stats['mean_degree']:.1f}")
    print(f"  Components: {stats['n_components']} (Giant Pct: {stats['giant_component_pct']:.1f}%)")

    print("\n[CLUSTERING]")
    levels = ms_result['levels']
    print(f"  Candidate Scales Extracted: {len(levels)}")
    for i, lvl in enumerate(levels):
        print(f"    - Scale {i}: res={lvl['resolution']:.4f}, K={lvl['n_clusters']}")

    print("\n[SELECTION]")
    selected = stability['selected_levels']
    print(f"  Selected Levels (indices): {selected}")
    for s_idx in selected:
        if s_idx < len(levels):
            print(f"    - Level {s_idx} -> K={levels[s_idx]['n_clusters']}")

    print("\n[MERGE]")
    print(f"  Total Merges: {merged['n_merges']}")
    print(f"  Merge Skipped: {'Yes' if merged['n_merges'] == 0 else 'No'}")

    print("\n[FINAL OUTPUT]")
    unique_ids = np.unique(fine_labels[fine_labels >= 0])
    counts = [np.sum(fine_labels == cid) for cid in unique_ids]
    singletons = np.sum(fine_labels == -1)
    print(f"  Final Clusters (Non-singleton): {len(unique_ids)}")
    if counts:
        print(f"  Sizes - Min: {np.min(counts)}, Max: {np.max(counts)}, Mean: {np.mean(counts):.1f}")
    else:
        print("  All clusters are singletons.")
    print(f"  Singleton Count: {singletons} ({singletons/N:.1%})")

    print("\n[CDLIB]")
    if cdlib_metrics:
        for k_met, v_met in cdlib_metrics.items():
            print(f"  {k_met}: {v_met:.4f}")
    else:
        print("  CDlib metrics unavailable/failed.")

if __name__ == "__main__":
    main()
