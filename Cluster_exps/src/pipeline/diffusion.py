"""
SGC-based feature diffusion for the EvoCluster merge stage.

Computes one-hop graph convolution on the pruned SNN graph to produce
topology-aware embeddings that blend local neighborhood structure into
the original PCA embedding space.

    X_diff  = D^{-1} @ (A + I) @ X
    X_alpha = (1 - alpha) * X + alpha * X_diff
"""

import logging

import numpy as np
from scipy import sparse


def compute_diffused_embeddings(X, adjacency, alpha=0.5, logger=None):
    """
    SGC-style one-hop feature diffusion on the SNN graph.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Original PCA-reduced, L2-normalized embeddings.
    adjacency : scipy.sparse matrix, shape (N, N)
        Pruned SNN adjacency (weighted or binary).
    alpha : float
        Mixing coefficient.  0 = pure original, 1 = pure diffused.

    Returns
    -------
    X_alpha : np.ndarray, shape (N, D)
        Mixed embeddings: (1 - alpha) * X + alpha * X_diff
    X_diff : np.ndarray, shape (N, D)
        Purely diffused embeddings (stored for potential reuse).
    """
    log = logger or logging.getLogger("multiscale")
    N = X.shape[0]

    # Â = A + I  (add self-loops)
    A_hat = adjacency.tocsr() + sparse.eye(N, format="csr")

    # Row-wise degree for normalization
    row_sums = np.asarray(A_hat.sum(axis=1)).ravel()
    inv_row_sums = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    D_inv = sparse.diags(inv_row_sums, format="csr")

    # One-hop diffusion: X_diff = D^{-1} @ Â @ X
    X_diff = (D_inv @ A_hat @ X).astype(np.float32)

    # Alpha mixing
    X_alpha = ((1.0 - alpha) * X + alpha * X_diff).astype(np.float32)

    log.info(
        f"  SGC diffusion: alpha={alpha:.2f}, "
        f"nnz(A)={adjacency.nnz}, N={N}, D={X.shape[1]}"
    )

    return X_alpha, X_diff
