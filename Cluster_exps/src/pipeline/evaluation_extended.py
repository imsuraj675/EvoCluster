"""
Extended evaluation metrics for EvoCluster.

Provides size-binned pairwise analysis, B-cubed precision/recall/F1,
split/merge error counts, and family-type classification (1:1 / 1:m / m:1).
"""

import logging
from collections import defaultdict, Counter

import numpy as np


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _comb2(n: int) -> int:
    return n * (n - 1) // 2


DEFAULT_SIZE_BINS = [
    ("singleton", 1, 1),
    ("small",     2, 5),
    ("medium",    6, 20),
    ("large",     21, 100),
    ("xlarge",    101, None),
]


# ---------------------------------------------------------------------------
#  Size-binned pairwise metrics
# ---------------------------------------------------------------------------

def compute_size_binned_metrics(
    true_labels,
    pred_labels,
    bins=None,
    logger=None,
):
    """
    Partition proteins by true orthogroup size, then compute pairwise P/R/F1
    restricted to proteins within each size bin.

    Parameters
    ----------
    true_labels : array-like of shape (N,)
        Ground-truth orthogroup assignments (string or int).
    pred_labels : array-like of shape (N,)
        Predicted cluster assignments (int, -1 = unassigned).
    bins : list of (name, lo, hi) tuples; hi=None means unbounded.

    Returns
    -------
    dict  bin_name -> {"precision", "recall", "f1", "n_proteins", "n_groups"}
    """
    log = logger or logging.getLogger("multiscale")
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)
    if bins is None:
        bins = DEFAULT_SIZE_BINS

    # build true group -> member indices
    group_members = defaultdict(list)
    for idx, t in enumerate(true_labels):
        group_members[t].append(idx)

    group_size = {g: len(m) for g, m in group_members.items()}

    results = {}
    for bin_name, lo, hi in bins:
        # collect protein indices that belong to groups in this size range
        indices = []
        n_groups = 0
        for g, members in group_members.items():
            sz = len(members)
            if sz < lo:
                continue
            if hi is not None and sz > hi:
                continue
            indices.extend(members)
            n_groups += 1

        if not indices:
            results[bin_name] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_proteins": 0, "n_groups": 0,
            }
            continue

        idx_arr = np.array(indices)
        sub_true = true_labels[idx_arr]
        sub_pred = pred_labels[idx_arr]

        # pairwise confusion restricted to this subset
        overlap_counts = defaultdict(int)
        cluster_counts = defaultdict(int)
        true_counts = defaultdict(int)

        for t, p in zip(sub_true, sub_pred):
            true_counts[t] += 1
            if int(p) >= 0:
                cluster_counts[int(p)] += 1
                overlap_counts[(int(p), t)] += 1

        tp = sum(_comb2(c) for c in overlap_counts.values())
        pred_pairs = sum(_comb2(c) for c in cluster_counts.values())
        true_pairs = sum(_comb2(c) for c in true_counts.values())

        fp = pred_pairs - tp
        fn = true_pairs - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[bin_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_proteins": len(indices),
            "n_groups": n_groups,
        }
        log.debug(
            f"  Size bin [{bin_name}] ({lo}-{hi or '∞'}): "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"({len(indices)} proteins, {n_groups} groups)"
        )

    return results


# ---------------------------------------------------------------------------
#  B-cubed metrics
# ---------------------------------------------------------------------------

def compute_bcubed_metrics(true_labels, pred_labels, logger=None):
    """
    Compute B-cubed precision, recall, and F1.

    B-cubed precision: for each item, fraction of its cluster-mates that
    share its true label.
    B-cubed recall: for each item, fraction of its true-group-mates that
    share its predicted cluster.

    Returns
    -------
    dict with "bcubed_precision", "bcubed_recall", "bcubed_f1"
    """
    log = logger or logging.getLogger("multiscale")
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)
    n = len(true_labels)

    if n == 0:
        return {"bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0}

    # Build cluster -> members and group -> members
    cluster_members = defaultdict(list)
    group_members = defaultdict(list)
    for i in range(n):
        if int(pred_labels[i]) >= 0:
            cluster_members[int(pred_labels[i])].append(i)
        group_members[true_labels[i]].append(i)

    # For each item, count same-true-label in its cluster (precision)
    # and same-cluster in its true group (recall)
    sum_prec = 0.0
    sum_rec = 0.0
    counted = 0

    # Pre-build per-cluster true-label counts
    cluster_true_counts = {}
    for cid, members in cluster_members.items():
        tc = Counter(true_labels[i] for i in members)
        cluster_true_counts[cid] = tc

    # Pre-build per-group cluster counts
    group_cluster_counts = {}
    for gid, members in group_members.items():
        gc = Counter(int(pred_labels[i]) for i in members if int(pred_labels[i]) >= 0)
        group_cluster_counts[gid] = gc

    for i in range(n):
        t = true_labels[i]
        p = int(pred_labels[i])

        # Precision: among items in my cluster, how many share my true label?
        if p >= 0 and p in cluster_true_counts:
            cluster_size = len(cluster_members[p])
            same_true_in_cluster = cluster_true_counts[p].get(t, 0)
            prec_i = same_true_in_cluster / cluster_size
        else:
            # Unassigned items: precision = 1 if singleton, else 0
            prec_i = 1.0  # singleton is trivially correct

        # Recall: among items in my true group, how many share my cluster?
        group_size = len(group_members[t])
        if p >= 0 and t in group_cluster_counts:
            same_cluster_in_group = group_cluster_counts[t].get(p, 0)
            rec_i = same_cluster_in_group / group_size
        else:
            rec_i = 1.0 / group_size  # only itself

        sum_prec += prec_i
        sum_rec += rec_i
        counted += 1

    bcubed_p = sum_prec / counted if counted > 0 else 0.0
    bcubed_r = sum_rec / counted if counted > 0 else 0.0
    bcubed_f1 = (
        2 * bcubed_p * bcubed_r / (bcubed_p + bcubed_r)
        if (bcubed_p + bcubed_r) > 0
        else 0.0
    )

    log.debug(f"  B-cubed: P={bcubed_p:.3f} R={bcubed_r:.3f} F1={bcubed_f1:.3f}")

    return {
        "bcubed_precision": float(bcubed_p),
        "bcubed_recall": float(bcubed_r),
        "bcubed_f1": float(bcubed_f1),
    }


# ---------------------------------------------------------------------------
#  Split / Merge error analysis
# ---------------------------------------------------------------------------

def compute_split_merge_errors(true_labels, pred_labels, logger=None):
    """
    Compute split and merge error summaries.

    Split error: true groups spread across >1 predicted cluster.
    Merge error: predicted clusters containing >1 true group.

    Returns
    -------
    dict with "n_split_groups", "n_merge_clusters", "split_details",
    "merge_details", "total_split_fragments", "total_merge_contaminants"
    """
    log = logger or logging.getLogger("multiscale")
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)

    # For each true group, count how many distinct clusters its members land in
    group_to_clusters = defaultdict(set)
    cluster_to_groups = defaultdict(set)

    for t, p in zip(true_labels, pred_labels):
        if int(p) >= 0:
            group_to_clusters[t].add(int(p))
            cluster_to_groups[int(p)].add(t)

    # Split analysis: true groups hitting >1 cluster
    split_groups = {g: clusters for g, clusters in group_to_clusters.items() if len(clusters) > 1}
    n_split = len(split_groups)
    total_split_fragments = sum(len(c) for c in split_groups.values())

    # Merge analysis: clusters containing >1 true group
    merge_clusters = {c: groups for c, groups in cluster_to_groups.items() if len(groups) > 1}
    n_merge = len(merge_clusters)
    total_merge_contaminants = sum(len(g) for g in merge_clusters.values())

    n_true_groups = len(group_to_clusters)
    n_pred_clusters = len(cluster_to_groups)

    log.debug(
        f"  Split/Merge: {n_split}/{n_true_groups} groups split, "
        f"{n_merge}/{n_pred_clusters} clusters impure"
    )

    return {
        "n_split_groups": n_split,
        "n_merge_clusters": n_merge,
        "n_true_groups": n_true_groups,
        "n_pred_clusters": n_pred_clusters,
        "total_split_fragments": total_split_fragments,
        "total_merge_contaminants": total_merge_contaminants,
        "split_ratio": n_split / max(n_true_groups, 1),
        "merge_ratio": n_merge / max(n_pred_clusters, 1),
        # Detailed per-group / per-cluster counts (top 20 worst)
        "worst_splits": sorted(
            [(g, len(c)) for g, c in split_groups.items()],
            key=lambda x: -x[1],
        )[:20],
        "worst_merges": sorted(
            [(c, len(g)) for c, g in merge_clusters.items()],
            key=lambda x: -x[1],
        )[:20],
    }


# ---------------------------------------------------------------------------
#  Family type classification
# ---------------------------------------------------------------------------

def classify_family_types(true_labels, pred_labels, logger=None):
    """
    Classify each true orthogroup by its mapping relationship:

    - 1:1 — maps to exactly one predicted cluster (and that cluster has only
             this group)
    - 1:m — split across multiple predicted clusters
    - m:1 — merged with other groups into one cluster

    Compute P/R/F1 for proteins belonging to each type.

    Returns
    -------
    dict with "types" (group -> type), "type_counts", and per-type P/R/F1
    """
    log = logger or logging.getLogger("multiscale")
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)

    # Map groups to clusters and clusters to groups
    group_to_clusters = defaultdict(set)
    cluster_to_groups = defaultdict(set)

    for t, p in zip(true_labels, pred_labels):
        if int(p) >= 0:
            group_to_clusters[t].add(int(p))
            cluster_to_groups[int(p)].add(t)

    # Classify each group
    group_type = {}
    for g, clusters in group_to_clusters.items():
        if len(clusters) == 1:
            c = next(iter(clusters))
            if len(cluster_to_groups.get(c, set())) == 1:
                group_type[g] = "1:1"
            else:
                group_type[g] = "m:1"
        else:
            group_type[g] = "1:m"

    # For groups that never appeared in a cluster (all members unassigned)
    all_groups = set(true_labels)
    for g in all_groups:
        if g not in group_type:
            group_type[g] = "unassigned"

    type_counts = Counter(group_type.values())
    log.debug(f"  Family types: {dict(type_counts)}")

    # Compute per-type pairwise metrics
    results = {"types": group_type, "type_counts": dict(type_counts)}

    # Build group membership
    group_members = defaultdict(list)
    for i, t in enumerate(true_labels):
        group_members[t].append(i)

    for ftype in ["1:1", "1:m", "m:1"]:
        indices = []
        for g, gt in group_type.items():
            if gt == ftype:
                indices.extend(group_members[g])

        if not indices:
            results[ftype] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_proteins": 0, "n_groups": type_counts.get(ftype, 0),
            }
            continue

        idx_arr = np.array(indices)
        sub_true = true_labels[idx_arr]
        sub_pred = pred_labels[idx_arr]

        overlap_counts = defaultdict(int)
        cluster_counts = defaultdict(int)
        true_counts = defaultdict(int)

        for t, p in zip(sub_true, sub_pred):
            true_counts[t] += 1
            if int(p) >= 0:
                cluster_counts[int(p)] += 1
                overlap_counts[(int(p), t)] += 1

        tp = sum(_comb2(c) for c in overlap_counts.values())
        pred_pairs = sum(_comb2(c) for c in cluster_counts.values())
        true_pairs = sum(_comb2(c) for c in true_counts.values())

        fp = pred_pairs - tp
        fn = true_pairs - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[ftype] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_proteins": len(indices),
            "n_groups": type_counts.get(ftype, 0),
        }
        log.debug(
            f"  Family type [{ftype}]: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"({len(indices)} prots, {type_counts.get(ftype, 0)} groups)"
        )

    return results


# ---------------------------------------------------------------------------
#  Convenience: run all extended metrics
# ---------------------------------------------------------------------------

def run_extended_evaluation(true_labels, pred_labels, logger=None):
    """Run all extended evaluation metrics and return a combined dict."""
    log = logger or logging.getLogger("multiscale")
    log.info("  Extended evaluation: size-binned, B-cubed, split/merge, family types")

    return {
        "size_binned": compute_size_binned_metrics(true_labels, pred_labels, logger=log),
        "bcubed": compute_bcubed_metrics(true_labels, pred_labels, logger=log),
        "split_merge": compute_split_merge_errors(true_labels, pred_labels, logger=log),
        "family_types": classify_family_types(true_labels, pred_labels, logger=log),
    }
