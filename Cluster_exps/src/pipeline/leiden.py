import logging
from collections import Counter
import numpy as np
import leidenalg as la

def _leiden_cluster_count(graph, resolution, objective="CPM"):
    weights = graph.es['weight'] if 'weight' in graph.edge_attributes() else None
    
    if objective.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition
        
    part = la.find_partition(
        graph, 
        partition_type, 
        weights=weights, 
        resolution_parameter=resolution, 
        n_iterations=-1
    )
    return len(set(part.membership))

def run_leiden_multiscale(
    snn_graph_result,
    *,
    resolution_grid=None,
    objective_function="CPM",
    beta=0.01,
    n_iterations=-1,
    min_cluster_size=1,
    seed=0,
    logger=None,
):
    from .evaluation import relabel_contiguous
    log = logger or logging.getLogger("multiscale")
    DEFAULT_RESOLUTIONS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    if resolution_grid is None:
        resolution_grid = DEFAULT_RESOLUTIONS

    g = snn_graph_result["graph"]
    N = snn_graph_result["n_nodes"]

    log.info(f"Step 3a: Running Leiden multiscale — {len(resolution_grid)} resolutions")

    levels = []
    label_matrix = np.zeros((N, len(resolution_grid)), dtype=np.int32)
    weights = g.es['weight'] if 'weight' in g.edge_attributes() else None

    if objective_function.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition

    for r_idx, res in enumerate(sorted(resolution_grid)):
        part = la.find_partition(
            g,
            partition_type,
            weights=weights,
            resolution_parameter=res,
            n_iterations=n_iterations,
            seed=seed
        )

        raw_labels = np.array(part.membership, dtype=np.int32)
        # Assuming enforce_min_cluster_size applies here (it used to), but min_cluster_size=1 is default so doing nothing
        labels, _ = relabel_contiguous(raw_labels)

        sizes = Counter(labels[labels >= 0].tolist())
        n_clusters = len(sizes)
        n_singletons = int(np.sum(labels == -1))

        if sizes:
            size_vals = list(sizes.values())
            min_s, max_s, med_s = min(size_vals), max(size_vals), int(np.median(size_vals))
        else:
            min_s = max_s = med_s = 0

        log.info(f"  res={res:.3f} → K={n_clusters} clusters (min={min_s}, max={max_s}, med={med_s}, singletons={n_singletons})")

        label_matrix[:, r_idx] = labels
        levels.append({
            "resolution": res,
            "labels": labels,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "sizes": sizes,
        })

    return {"levels": levels, "label_matrix": label_matrix, "resolution_grid": sorted(resolution_grid)}

def build_cluster_hierarchy(multiscale_result, *, logger=None):
    log = logger or logging.getLogger("multiscale")
    log.info("Step 3b: Building cluster hierarchy (argmax overlap, no threshold)")

    levels = multiscale_result["levels"]
    n_levels = len(levels)

    tree = {}
    overlap_strengths = {}
    nodes = []
    edges = []

    for lev_idx, lev in enumerate(levels):
        for cid in sorted(lev["sizes"].keys()):
            nodes.append((lev_idx, cid))

    min_overlap = 1.0
    for lev_idx in range(1, n_levels):
        fine_labels = levels[lev_idx]["labels"]
        coarse_labels = levels[lev_idx - 1]["labels"]
        fine_clusters = sorted(levels[lev_idx]["sizes"].keys())

        for fine_cid in fine_clusters:
            fine_mask = fine_labels == fine_cid
            if fine_mask.sum() == 0:
                continue
            coarse_of_fine = coarse_labels[fine_mask]
            coarse_of_fine = coarse_of_fine[coarse_of_fine >= 0]
            if len(coarse_of_fine) == 0:
                continue

            counter = Counter(coarse_of_fine.tolist())
            best_coarse, best_count = counter.most_common(1)[0]
            overlap_frac = best_count / fine_mask.sum()

            child_node = (lev_idx, fine_cid)
            parent_node = (lev_idx - 1, best_coarse)

            tree[child_node] = parent_node
            overlap_strengths[(parent_node, child_node)] = overlap_frac
            edges.append((parent_node, child_node, overlap_frac))
            min_overlap = min(min_overlap, overlap_frac)

    roots = [(0, cid) for cid in sorted(levels[0]["sizes"].keys())] if n_levels > 0 else []
    leaves = [(n_levels - 1, cid) for cid in sorted(levels[-1]["sizes"].keys())] if n_levels > 0 else []

    log.info(f"  {len(nodes)} nodes, {len(edges)} parent-child links")
    log.info(f"  Weakest parent-child overlap: {min_overlap:.3f}")

    return {
        "nodes": nodes,
        "edges": edges,
        "roots": roots,
        "leaves": leaves,
        "tree": tree,
        "overlap_strengths": overlap_strengths,
        "n_levels": n_levels,
    }
