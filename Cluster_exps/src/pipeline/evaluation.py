import numpy as np
import logging
from sklearn.metrics import adjusted_mutual_info_score

EPS = 1e-12

def relabel_contiguous(labels: np.ndarray):
    """Map labels >= 0 to 0..K-1, keep -1 as -1."""
    labels = labels.copy()
    kept = np.unique(labels[labels >= 0])
    remap = {old: new for new, old in enumerate(kept.tolist())}
    for i in range(labels.size):
        if labels[i] >= 0:
            labels[i] = remap[labels[i]]
    return labels, kept

def compute_centroids_from_labels(X: np.ndarray, labels: np.ndarray):
    """Return (K, D) centroids aligned with contiguous labels 0..K-1."""
    cluster_ids = np.unique(labels[labels >= 0])
    if cluster_ids.size == 0:
        return None
    K = cluster_ids.max() + 1
    D = X.shape[1]
    centroids = np.zeros((K, D), dtype=np.float64)
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        centroids[int(c)] = X[idx].mean(axis=0)
    return centroids

def compute_precision_recall_f1(n_correct, n_retrieved, n_true):
    precision = 0.0 if n_retrieved == 0 else n_correct / n_retrieved
    recall = 0.0 if n_true == 0 else n_correct / n_true
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def compute_cluster_distances_cosine(X, labels, centroids=None, eps=EPS):
    """Returns (N, K) cosine-distance matrix: D[i,j] = 1 - cos(x_i, center_j)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    cluster_ids = np.unique(labels[labels >= 0])
    k = len(cluster_ids)
    if k == 0:
        return np.full((n, 0), np.inf)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    D = np.full((n, k), np.inf, dtype=np.float64)
    for j, clu in enumerate(cluster_ids):
        idx = np.where(labels == clu)[0]
        center = X[idx].mean(axis=0) if centroids is None else np.asarray(centroids[int(clu)], dtype=np.float64)
        center_n = center / (np.linalg.norm(center) + eps)
        D[:, j] = 1.0 - Xn @ center_n
    return D

def get_total_counts(prot_meta):
    """Compute total true n:m pairs and 1:1 groups."""
    species = list(set(p[0] for p in prot_meta))
    ogs = list(set(p[4] for p in prot_meta))
    total_n2m = 0
    total_121 = 0
    for og in ogs:
        members = [p[0] for p in prot_meta if p[4] == og]
        counts = [members.count(s) for s in species]
        for i in range(len(counts)):
            for j in range(i + 1, len(counts)):
                total_n2m += counts[i] * counts[j]
        if len(members) == len(species) and len(set(members)) == len(species):
            total_121 += 1
    return total_n2m, total_121

def measure_pairwise_from_labels(labels, prot_meta, X):
    from scipy.spatial.distance import pdist, squareform
    clusters = sorted(set(int(l) for l in labels if l >= 0))
    orth_naive, orth_dist, orth_121 = [], [], []
    species_all = sorted(set(p[0] for p in prot_meta))
    n_species = len(species_all)

    for c_idx, c in enumerate(clusters):
        ind = [i for i, l in enumerate(labels) if l == c]
        if len(ind) <= 1:
            continue
            
        prots = [prot_meta[i] for i in ind]
        specs = [p[0] for p in prots]
        unique_specs = sorted(list(set(specs)))
        
        # 1. Naive pairs (all pairs across different species)
        if len(unique_specs) > 1:
            for i in range(len(specs)):
                for j in range(i + 1, len(specs)):
                    if specs[i] != specs[j]:
                        orth_naive.append([prots[i], prots[j]])
                        
        # 2. Distance-based pairs (minimum pairwise distance per species-pair)
        if len(unique_specs) > 1:
            X_clu = X[ind]
            dist_mat = squareform(pdist(X_clu, metric='cosine'))
            for i_spec in range(len(unique_specs)):
                for j_spec in range(i_spec + 1, len(unique_specs)):
                    spec_a = unique_specs[i_spec]
                    spec_b = unique_specs[j_spec]
                    
                    idx_a = [i for i, s in enumerate(specs) if s == spec_a]
                    idx_b = [i for i, s in enumerate(specs) if s == spec_b]
                    
                    min_dist = float('inf')
                    best_pair = None
                    for a in idx_a:
                        for b in idx_b:
                            if dist_mat[a, b] < min_dist:
                                min_dist = dist_mat[a, b]
                                best_pair = [prots[a], prots[b]]
                    
                    if best_pair is not None:
                        orth_dist.append(best_pair)
                        
        # 3. 1:1 orthogroups
        if len(specs) == n_species and len(unique_specs) == n_species:
            orth_121.append(prots)
            
    return orth_naive, orth_dist, orth_121

def evaluate_run(labels, prot_meta, X, total_n2m, total_121):
    """Full evaluation: naive/distance/121 P/R/F1 + AMI."""
    orth_naive, orth_dist, orth_121 = measure_pairwise_from_labels(labels, prot_meta, X)

    def count_correct_pairs(pair_list):
        c = 0
        for pair in pair_list:
            if isinstance(pair[0], (list, tuple)):
                if pair[0][4] == pair[1][4]:
                    c += 1
        return c

    n_correct_nv = count_correct_pairs(orth_naive)
    n_correct_dist = count_correct_pairs(orth_dist)
    n_correct_121 = sum(1 for grp in orth_121 if len(set(p[4] for p in grp)) == 1)

    p_nv, r_nv, f1_nv = compute_precision_recall_f1(n_correct_nv, len(orth_naive), total_n2m)
    p_dist, r_dist, f1_dist = compute_precision_recall_f1(n_correct_dist, len(orth_dist), total_n2m)
    p_121, r_121, f1_121 = compute_precision_recall_f1(n_correct_121, len(orth_121), total_121)

    ami = float(adjusted_mutual_info_score(
        [p[4] for p in prot_meta], labels
    ))

    return {
        "naive": (p_nv, r_nv, f1_nv, n_correct_nv, len(orth_naive)),
        "distance": (p_dist, r_dist, f1_dist, n_correct_dist, len(orth_dist)),
        "121": (p_121, r_121, f1_121, n_correct_121, len(orth_121)),
        "AMI": ami,
    }

def evaluate_clustering(labels, prot_meta, X_pca, total_n2m, total_121, *, graph=None, alpha=0.5, level_name="", logger=None):
    """Full evaluation: P/R/F1 + AMI + combined score + Topological metrics."""
    log = logger or logging.getLogger("multiscale")

    n_clusters = len(set(int(l) for l in labels if l >= 0))
    n_singletons = int(np.sum(labels == -1))
    log.info(f"Step 6: Evaluating [{level_name}] — {n_clusters} clusters, {n_singletons} singletons")

    metrics = evaluate_run(labels, prot_meta, X_pca, total_n2m, total_121)
    combined = alpha * metrics["distance"][2] + (1 - alpha) * metrics["121"][2]

    log.info(f"  naive  P={metrics['naive'][0]:.3f}  R={metrics['naive'][1]:.3f}  F1={metrics['naive'][2]:.3f}")
    log.info(f"  dist   P={metrics['distance'][0]:.3f}  R={metrics['distance'][1]:.3f}  F1={metrics['distance'][2]:.3f}")
    log.info(f"  1:1    P={metrics['121'][0]:.3f}  R={metrics['121'][1]:.3f}  F1={metrics['121'][2]:.3f}")
    
    # Run Topological metrics if graph was provided
    if graph is not None:
        try:
            from .evaluation_cdlib import evaluate_topology_cdlib
            cdlib_metrics = evaluate_topology_cdlib(graph, labels, logger=log)
            if cdlib_metrics:
                metrics.update(cdlib_metrics)
                mod = cdlib_metrics.get('modularity', float('nan'))
                cond = cdlib_metrics.get('conductance', float('nan'))
                log.info(f"  AMI={metrics['AMI']:.4f}  Combined={combined:.4f} | Modularity={mod:.4f} Conductance={cond:.4f}")
            else:
                log.info(f"  AMI={metrics['AMI']:.4f}  Combined={combined:.4f}")
        except Exception as e:
            log.debug(f"Topological metric integration failed: {e}")
            log.info(f"  AMI={metrics['AMI']:.4f}  Combined={combined:.4f}")
    else:
        log.info(f"  AMI={metrics['AMI']:.4f}  Combined={combined:.4f}")

    metrics["combined_score"] = combined
    metrics["n_clusters"] = n_clusters
    metrics["n_singletons"] = n_singletons
    return metrics
