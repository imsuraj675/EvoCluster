import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


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
    k = cluster_ids.max() + 1
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        centroids[int(c)] = X[idx].mean(axis=0)
    return centroids


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def extract_true_labels(prot_meta):
    return np.asarray([p[4] for p in prot_meta], dtype=object)


def get_ground_truth_stats(prot_meta):
    true_labels = extract_true_labels(prot_meta)
    true_counts = defaultdict(int)
    for label in true_labels:
        true_counts[label] += 1

    return {
        "n_proteins": int(len(true_labels)),
        "n_orthogroups": int(len(true_counts)),
        "true_positive_pairs": int(sum(_comb2(n) for n in true_counts.values())),
    }


def compute_pairwise_confusion_metrics(true_labels, pred_labels):
    """
    Compute pairwise TP/FP/FN/TN for clustering labels in O(N).

    Any predicted label < 0 is treated as unassigned/singleton, so it contributes
    no predicted-positive pairs. This matches the intended semantics for -1 labels.
    """
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)

    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError("true_labels and pred_labels must have the same length")

    n = int(true_labels.shape[0])

    cluster_counts = defaultdict(int)
    true_counts = defaultdict(int)
    overlap_counts = defaultdict(int)

    for t_label, p_label in zip(true_labels, pred_labels):
        true_counts[t_label] += 1
        if int(p_label) >= 0:
            cluster_counts[int(p_label)] += 1
            overlap_counts[(int(p_label), t_label)] += 1

    tp = sum(_comb2(count) for count in overlap_counts.values())
    pred_pairs = sum(_comb2(count) for count in cluster_counts.values())
    true_pairs = sum(_comb2(count) for count in true_counts.values())

    fp = pred_pairs - tp
    fn = true_pairs - tp
    total_pairs = _comb2(n)
    tn = total_pairs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "pred_positive_pairs": int(pred_pairs),
        "true_positive_pairs": int(true_pairs),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_run(labels, prot_meta):
    """Primary evaluation: pairwise precision/recall/F1 + AMI."""
    true_labels = extract_true_labels(prot_meta)
    pairwise = compute_pairwise_confusion_metrics(true_labels, labels)
    ami = float(adjusted_mutual_info_score(true_labels, labels))

    return {
        "pairwise": pairwise,
        "AMI": ami,
    }


def evaluate_clustering(labels, prot_meta, *, graph=None, level_name="", extended_eval=False, logger=None):
    """Evaluate clustering against true orthogroup labels."""
    log = logger or logging.getLogger("multiscale")

    n_clusters = len(set(int(l) for l in labels if l >= 0))
    n_singletons = int(np.sum(labels == -1))
    log.info(f"Step 6: Evaluating [{level_name}] — {n_clusters} clusters, {n_singletons} singletons")

    metrics = evaluate_run(labels, prot_meta)
    pairwise = metrics["pairwise"]

    log.info(
        f"  pairwise  P={pairwise['precision']:.3f}  "
        f"R={pairwise['recall']:.3f}  F1={pairwise['f1']:.3f}"
    )
    log.info(
        f"  pairwise  TP={pairwise['TP']}  FP={pairwise['FP']}  "
        f"FN={pairwise['FN']}  TN={pairwise['TN']}"
    )

    if graph is not None:
        try:
            from .evaluation_cdlib import evaluate_topology_cdlib

            cdlib_metrics = evaluate_topology_cdlib(graph, labels, logger=log)
            if cdlib_metrics:
                metrics.update(cdlib_metrics)
                mod = cdlib_metrics.get("modularity", float("nan"))
                cond = cdlib_metrics.get("conductance", float("nan"))
                log.info(
                    f"  AMI={metrics['AMI']:.4f}  PairwiseF1={pairwise['f1']:.4f} | "
                    f"Modularity={mod:.4f} Conductance={cond:.4f}"
                )
            else:
                log.info(f"  AMI={metrics['AMI']:.4f}  PairwiseF1={pairwise['f1']:.4f}")
        except Exception as e:
            log.debug(f"Topological metric integration failed: {e}")
            log.info(f"  AMI={metrics['AMI']:.4f}  PairwiseF1={pairwise['f1']:.4f}")
    else:
        log.info(f"  AMI={metrics['AMI']:.4f}  PairwiseF1={pairwise['f1']:.4f}")

    # ── Extended evaluation ──
    if extended_eval:
        try:
            from .evaluation_extended import run_extended_evaluation
            true_labels = extract_true_labels(prot_meta)
            ext = run_extended_evaluation(true_labels, labels, logger=log)
            metrics["extended"] = ext

            # Log headline extended metrics
            bcubed = ext.get("bcubed", {})
            sm = ext.get("split_merge", {})
            log.info(
                f"  B-cubed  P={bcubed.get('bcubed_precision', 0):.3f}  "
                f"R={bcubed.get('bcubed_recall', 0):.3f}  "
                f"F1={bcubed.get('bcubed_f1', 0):.3f}"
            )
            log.info(
                f"  Split/Merge: {sm.get('n_split_groups', 0)} groups split, "
                f"{sm.get('n_merge_clusters', 0)} clusters impure"
            )

            # Log size-binned summary
            size_bins = ext.get("size_binned", {})
            for bin_name, bm in size_bins.items():
                if bm.get("n_proteins", 0) > 0:
                    log.info(
                        f"  [{bin_name}] P={bm['precision']:.3f} R={bm['recall']:.3f} "
                        f"F1={bm['f1']:.3f} ({bm['n_proteins']} prots, {bm['n_groups']} groups)"
                    )
        except Exception as e:
            log.debug(f"Extended evaluation failed: {e}")

    metrics["primary_score"] = pairwise["f1"]
    metrics["combined_score"] = pairwise["f1"]
    metrics["n_clusters"] = n_clusters
    metrics["n_singletons"] = n_singletons
    return metrics

