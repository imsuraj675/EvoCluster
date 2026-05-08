# cluster_lib.py
"""
Clustering helpers for EvoCluster experiments.

Public API:
  - l2_normalize(X)
  - _knn_edges(X, n_neighbors, ...)  -> edges, weights
  - build_knn_connectivity(X, n_neighbors, ...) -> CSR adjacency
"""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


# -------------------------
# Basic utilities
# -------------------------
def l2_normalize(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)


# -------------------------
# kNN graph construction
# -------------------------
def _knn_edges(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "cosine",
    use_faiss: bool = True,
    faiss_M: int = 32,
    faiss_ef_search: int = 64,
    faiss_ef_construction: int = 200,
    mutual_knn: bool = True,
    eps_w: float = 1e-6,
):
    """
    Return undirected edges list and positive weights for a kNN graph.
    edges: list of (u,v) with u<v
    weights: list of floats (cosine similarity mapped to [0,1])
    """
    X = np.asarray(X)
    N, D = X.shape

    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    X_use = l2_normalize(X) if metric == "cosine" else X

    # try FAISS HNSW
    nbrs = None
    vals = None
    if use_faiss:
        try:
            import faiss
            Xf = np.ascontiguousarray(X_use.astype(np.float32))
            if metric == "cosine":
                idx = faiss.IndexHNSWFlat(D, faiss_M, faiss.METRIC_INNER_PRODUCT)
            else:
                idx = faiss.IndexHNSWFlat(D, faiss_M, faiss.METRIC_L2)
            idx.hnsw.efConstruction = faiss_ef_construction
            idx.hnsw.efSearch = faiss_ef_search
            idx.add(Xf)
            scores_or_d2, I = idx.search(Xf, n_neighbors + 1)
            nbrs = []
            vals = []
            for i in range(N):
                js = I[i]
                vs = scores_or_d2[i]
                keep = [(int(j), float(v)) for j, v in zip(js, vs) if j >= 0 and j != i]
                keep = keep[:n_neighbors]
                nbrs.append([j for j, _ in keep])
                vals.append([v for _, v in keep])
        except Exception:
            use_faiss = False

    # fallback: exact sklearn kNN
    if not use_faiss:
        nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric=("cosine" if metric == "cosine" else "euclidean"),
        )
        nn.fit(X_use)
        dist, I = nn.kneighbors(X_use, return_distance=True)
        nbrs = []
        vals = []
        for i in range(N):
            keep = [(int(j), float(d)) for j, d in zip(I[i], dist[i]) if j != i]
            keep = keep[:n_neighbors]
            nbrs.append([j for j, _ in keep])
            vals.append([v for _, v in keep])

        if metric == "euclidean":
            vals = [[v * v for v in row] for row in vals]

    # convert neighbor vals -> positive weights
    if metric == "cosine":
        def sim_to_w(sim):
            w = (sim + 1.0) * 0.5
            return float(max(w, eps_w))
        wvals = [[sim_to_w(s) for s in row] for row in vals]
    else:
        flat = np.array([v for row in vals for v in row], dtype=np.float64)
        sigma2 = np.median(flat) if flat.size else 1.0
        sigma2 = max(sigma2, 1e-12)
        def d2_to_w(d2):
            w = np.exp(-d2 / (2.0 * sigma2))
            return float(max(w, eps_w))
        wvals = [[d2_to_w(d2) for d2 in row] for row in vals]

    # build undirected edge set
    edge_w = {}
    if mutual_knn:
        nbr_sets = [set(row) for row in nbrs]
        for i in range(N):
            for j, w in zip(nbrs[i], wvals[i]):
                if i in nbr_sets[j]:
                    u, v = (i, j) if i < j else (j, i)
                    edge_w[(u, v)] = max(edge_w.get((u, v), 0.0), w)
    else:
        for i in range(N):
            for j, w in zip(nbrs[i], wvals[i]):
                u, v = (i, j) if i < j else (j, i)
                if u != v:
                    edge_w[(u, v)] = max(edge_w.get((u, v), 0.0), w)

    edges = list(edge_w.keys())
    weights = [edge_w[e] for e in edges]
    return edges, weights


def build_knn_connectivity(
    X: np.ndarray,
    n_neighbors: int = 50,
    metric: str = "cosine",
    use_faiss: bool = True,
    faiss_M: int = 32,
    faiss_ef_search: int = 64,
    faiss_ef_construction: int = 200,
):
    """
    Returns symmetric CSR adjacency (N,N) with 1 where kNN edges exist.
    Used as connectivity constraint for AgglomerativeClustering.
    """
    X_use = X if metric != "cosine" else l2_normalize(X)

    # try FAISS HNSW
    try:
        if use_faiss:
            import faiss
            Xf = np.ascontiguousarray(X_use.astype(np.float32))
            d = Xf.shape[1]
            if metric == "cosine":
                index = faiss.IndexHNSWFlat(d, faiss_M, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexHNSWFlat(d, faiss_M, faiss.METRIC_L2)
            index.hnsw.efConstruction = faiss_ef_construction
            index.hnsw.efSearch = faiss_ef_search
            index.add(Xf)
            _, I = index.search(Xf, n_neighbors + 1)
            rows, cols = [], []
            n = Xf.shape[0]
            for i in range(n):
                for j in I[i]:
                    if j < 0 or j == i:
                        continue
                    rows.append(i)
                    cols.append(int(j))
            A = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
            )
            A = A.maximum(A.T)
            return A
    except Exception:
        pass

    # exact fallback
    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=("cosine" if metric == "cosine" else "euclidean"),
    )
    nn.fit(X_use)
    _, indices = nn.kneighbors(X_use, return_distance=True)
    rows, cols = [], []
    n = X_use.shape[0]
    for i in range(n):
        for j in indices[i]:
            if j == i:
                continue
            rows.append(i)
            cols.append(int(j))
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    A = A.maximum(A.T)
    return A