import math
import time
import logging
import numpy as np
import torch
from scipy import sparse

def compute_adaptive_k(n_samples: int, coeff: float = 0.4, cap: int = 150) -> int:
    return min(math.ceil(coeff * math.sqrt(n_samples)), cap)

def build_candidate_neighbors(X, *, k_candidates, use_faiss=True, use_gpu=False, logger=None):
    log = logger or logging.getLogger("multiscale")
    log.info(f"Step 2a: Building candidate neighbors — k={k_candidates}, faiss={use_faiss}, gpu={use_gpu}")
    N, D = X.shape
    Xf = np.ascontiguousarray(X.astype(np.float32))
    indices = None
    scores = None
    backend_used = "sklearn"
    if use_faiss:
        try:
            import faiss
            gpu_faiss = False
            if use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index_cpu = faiss.IndexFlatIP(D)
                    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                    gpu_faiss = True
                    backend_used = "FAISS GPU (IndexFlatIP)"
                except Exception as e_gpu:
                    log.debug(f"  FAISS GPU unavailable ({e_gpu}), falling back to CPU FAISS")
                    gpu_faiss = False
            if not gpu_faiss:
                index = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = max(64, k_candidates + 16)
                backend_used = "FAISS CPU (IndexHNSWFlat)"
            index.add(Xf)
            scores, indices = index.search(Xf, k_candidates + 1)
        except Exception as e:
            log.debug(f"  FAISS unavailable ({e}), falling back to sklearn")
            use_faiss = False
            backend_used = "sklearn"
    if not use_faiss:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_candidates + 1, metric="cosine")
        nn.fit(X)
        dists, indices = nn.kneighbors(X)
        scores = 1.0 - dists
        backend_used = "sklearn NearestNeighbors"
    log.info(f"  Backend: {backend_used}")
    nbr_indices = []
    nbr_scores = []
    for i in range(N):
        js = indices[i]
        vs = scores[i]
        keep = [(int(j), float(v)) for j, v in zip(js, vs) if j >= 0 and j != i]
        keep = keep[:k_candidates]
        nbr_indices.append([j for j, _ in keep])
        nbr_scores.append([v for _, v in keep])
    neighbor_sets = [set(row) for row in nbr_indices]
    total_edges = sum(len(row) for row in nbr_indices)
    log.info(f"  Found {total_edges} directed neighbor links")
    return {
        "indices": nbr_indices,
        "scores": nbr_scores,
        "neighbor_sets": neighbor_sets,
        "k_candidates": k_candidates,
        "N": N,
    }

def build_snn_graph(
    candidate_neighbors,
    *,
    snn_k=None,
    prune_method="inverse_k",
    prune_percentile=30.0,
    min_giant_component_pct=95.0,
    logger=None,
):
    import igraph as ig
    log = logger or logging.getLogger("multiscale")
    N = candidate_neighbors["N"]
    nbr_indices = candidate_neighbors["indices"]
    neighbor_sets = candidate_neighbors["neighbor_sets"]
    k_cand = candidate_neighbors["k_candidates"]
    k = snn_k if snn_k is not None else k_cand
    log.info(f"Step 2b: Building SNN graph — Jaccard, prune={prune_method}, snn_k={k}")
    if k < k_cand:
        neighbor_sets = [set(nbr_indices[i][:k]) for i in range(N)]
    edge_weights = {}
    for i in range(N):
        for j in nbr_indices[i][:k]:
            if j <= i:
                continue
            if i not in neighbor_sets[j]:
                continue
            intersection = len(neighbor_sets[i] & neighbor_sets[j])
            union = len(neighbor_sets[i] | neighbor_sets[j])
            if union == 0:
                continue
            jaccard = intersection / union
            edge_weights[(i, j)] = jaccard
    edges_before_prune = len(edge_weights)
    if prune_method == "inverse_k":
        threshold = 1.0 / k
    elif prune_method == "percentile":
        if edge_weights:
            all_w = np.array(list(edge_weights.values()))
            threshold = np.percentile(all_w, prune_percentile)
        else:
            threshold = 0.0
    else:
        threshold = 0.0
    pruned = {e: w for e, w in edge_weights.items() if w >= threshold}
    edges_after_prune = len(pruned)
    log.info(f"  Prune threshold: {threshold:.4f}")
    log.info(f"  Edges: {edges_before_prune} → {edges_after_prune} after pruning")
    if edges_before_prune > 0:
        prune_pct = 100.0 * (1 - edges_after_prune / edges_before_prune)
        if prune_pct > 60:
            log.warning(f"  ⚠ HEAVY PRUNING: {prune_pct:.0f}% of edges removed!")
        elif prune_pct > 40:
            log.info(f"  ℹ Moderate pruning ({prune_pct:.0f}% removed).")
    if edges_after_prune == 0:
        log.warning(f"  ⚠ ZERO EDGES after pruning! Graph is empty.")
    edge_list = list(pruned.keys())
    weight_list = list(pruned.values())
    g = ig.Graph(n=N, edges=edge_list, directed=False)
    g.es["weight"] = weight_list
    components = g.connected_components()
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    giant_pct = 100.0 * comp_sizes[0] / N if N > 0 else 0.0
    mean_degree = 2.0 * g.ecount() / N if N > 0 else 0.0
    log.info(f"  Giant component: {giant_pct:.1f}% ({len(comp_sizes)} components)")
    log.info(f"  Mean degree: {mean_degree:.1f}")
    if giant_pct < min_giant_component_pct:
        log.warning(f"  ⚠ WARNING: Giant component {giant_pct:.1f}% < {min_giant_component_pct}%!")
    if mean_degree < 5:
        log.warning(f"  ⚠ VERY SPARSE GRAPH: mean degree {mean_degree:.1f} < 5.")
    elif mean_degree > 200:
        log.warning(f"  ⚠ VERY DENSE GRAPH: mean degree {mean_degree:.1f} > 200.")
    rows = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    cols = [e[1] for e in edge_list] + [e[0] for e in edge_list]
    data = weight_list + weight_list
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return {
        "graph": g,
        "adjacency": adjacency,
        "n_edges": g.ecount(),
        "n_nodes": N,
        "stats": {
            "giant_component_pct": giant_pct,
            "mean_degree": mean_degree,
            "n_components": len(comp_sizes),
            "edges_before_prune": edges_before_prune,
            "edges_after_prune": edges_after_prune,
            "prune_threshold_used": threshold,
        },
    }

def run_phate_scale_discovery(X_pca, *, n_jobs=-1, random_state=0, logger=None):
    log = logger or logging.getLogger("multiscale")
    log.info("Step 2c: Running Multiscale PHATE scale discovery")
    t0 = time.time()
    try:
        import multiscale_phate as mphate
    except ImportError:
        raise ImportError("multiscale-phate is required when --use_phate is set")
    mp = mphate.Multiscale_PHATE(n_pca=None, n_jobs=n_jobs, random_state=random_state)
    mp.fit(X_pca)
    levels = mp.levels
    gradient = mp.gradient
    NxTs = mp.NxTs
    cluster_counts = []
    for lev in levels:
        n_clusters = len(set(NxTs[lev]))
        cluster_counts.append(n_clusters)
    elapsed = time.time() - t0
    log.info(f"  PHATE found {len(levels)} stable levels in {elapsed:.1f}s")
    for i, (lev, kc) in enumerate(zip(levels, cluster_counts)):
        log.info(f"    level {i}: condensation_iter={lev}, K={kc}")
    if len(levels) < 2:
        log.warning("  ⚠ PHATE found fewer than 2 stable levels.")
    if len(levels) > 10:
        log.info(f"  ℹ PHATE found {len(levels)} levels — will use top 6-8 by spread.")
    return {
        "levels": levels,
        "gradient": gradient,
        "cluster_counts": cluster_counts,
        "NxTs": NxTs,
        "elapsed": elapsed,
    }

def map_phate_to_leiden_resolutions(
    snn_graph, target_counts, *,
    objective="CPM",
    search_lo=1e-4, search_hi=5.0,
    tolerance=0.20,
    max_iter=25,
    logger=None,
):
    from .leiden import _leiden_cluster_count
    log = logger or logging.getLogger("multiscale")
    log.info(f"Step 2d: Mapping {len(target_counts)} PHATE levels → Leiden resolutions")
    g = snn_graph["graph"]
    found_resolutions = []
    for target_k in sorted(set(target_counts), reverse=True):
        lo, hi = search_lo, search_hi
        best_r, best_diff = (lo + hi) / 2, float("inf")
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            k = _leiden_cluster_count(g, mid, objective)
            diff = abs(k - target_k) / max(target_k, 1)
            if diff < best_diff:
                best_diff = diff
                best_r = mid
            if diff <= tolerance:
                break
            if k < target_k:
                lo = mid
            else:
                hi = mid
        found_resolutions.append(best_r)
        log.debug(f"    target K={target_k} → res={best_r:.6f} (got K within {best_diff*100:.0f}%)")
    derived = sorted(set(round(r, 6) for r in found_resolutions))
    if len(derived) < 2:
        log.warning("  ⚠ PHATE mapping produced <2 distinct resolutions, adding boundary values.")
        derived = sorted(set(derived + [search_lo, (search_lo + search_hi) / 10]))
    log.info(f"  Derived resolution grid ({len(derived)} values): {[f'{r:.4f}' for r in derived]}")
    return derived
