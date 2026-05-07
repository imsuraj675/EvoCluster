import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, homogeneity_score, completeness_score


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


def extract_species_labels(prot_meta):
    """Extract species prefix from prot_meta (first pipe-delimited field)."""
    return np.asarray([p[0].lstrip(">").strip() for p in prot_meta], dtype=object)


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


def compute_orthogroup_type_metrics(true_labels, pred_labels, species_labels, logger=None):
    """
    Classify each true orthogroup by species composition and compute
    pairwise P/R/F1 restricted to proteins in each type.

    Types:
      1:1  — every species in the orthogroup has exactly 1 gene
      1:m  — mix of species with 1 gene and species with >1 gene
      n:m  — every species in the orthogroup has >1 gene
    """
    log = logger or logging.getLogger("multiscale")
    true_labels = np.asarray(true_labels, dtype=object)
    pred_labels = np.asarray(pred_labels)
    species_labels = np.asarray(species_labels, dtype=object)

    # Build per-orthogroup species-count profile
    group_members = defaultdict(list)
    group_species_counts = defaultdict(lambda: defaultdict(int))
    for i, (t, sp) in enumerate(zip(true_labels, species_labels)):
        group_members[t].append(i)
        group_species_counts[t][sp] += 1

    # Classify each orthogroup
    group_type = {}
    for g, sp_counts in group_species_counts.items():
        counts = list(sp_counts.values())
        if all(c == 1 for c in counts):
            group_type[g] = "1:1"
        elif all(c > 1 for c in counts):
            group_type[g] = "n:m"
        else:
            group_type[g] = "1:m"

    from collections import Counter
    type_counts = Counter(group_type.values())
    log.info(
        f"  orthogroup types: "
        f"1:1={type_counts.get('1:1', 0)}  "
        f"1:m={type_counts.get('1:m', 0)}  "
        f"n:m={type_counts.get('n:m', 0)}"
    )

    results = {"type_counts": dict(type_counts)}

    for ftype in ["1:1", "1:m", "n:m"]:
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

        overlap = defaultdict(int)
        cluster_c = defaultdict(int)
        true_c = defaultdict(int)

        for t, p in zip(sub_true, sub_pred):
            true_c[t] += 1
            if int(p) >= 0:
                cluster_c[int(p)] += 1
                overlap[(int(p), t)] += 1

        tp = sum(_comb2(c) for c in overlap.values())
        pred_pairs = sum(_comb2(c) for c in cluster_c.values())
        true_pairs = sum(_comb2(c) for c in true_c.values())
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
        log.info(
            f"  [{ftype}]  P={precision:.3f}  R={recall:.3f}  "
            f"F1={f1:.3f}  ({len(indices)} prots, {type_counts.get(ftype, 0)} groups)"
        )

    return results


def compute_per_species_diagnostics(pred_labels, prot_meta, logger=None):
    """Compute per-species assignment rates to diagnose species-specific recall failures."""
    log = logger or logging.getLogger("multiscale")
    species = extract_species_labels(prot_meta)
    pred_labels = np.asarray(pred_labels)
    results = {}
    for sp in sorted(set(species.tolist())):
        mask = species == sp
        total = int(mask.sum())
        assigned = int(np.sum(pred_labels[mask] >= 0))
        results[sp] = {
            "total": total,
            "assigned": assigned,
            "unassigned": total - assigned,
            "singleton_rate": (total - assigned) / max(total, 1),
        }
        log.info(
            f"  [{sp}] {assigned}/{total} assigned "
            f"({100 * assigned / max(total, 1):.1f}%), "
            f"{total - assigned} singletons"
        )
    return results


def evaluate_run(labels, prot_meta):
    """Primary evaluation: pairwise precision/recall/F1 + AMI + V-measure."""
    true_labels = extract_true_labels(prot_meta)
    pairwise = compute_pairwise_confusion_metrics(true_labels, labels)
    ami = float(adjusted_mutual_info_score(true_labels, labels))
    v_measure = float(v_measure_score(true_labels, labels))
    homogeneity = float(homogeneity_score(true_labels, labels))
    completeness = float(completeness_score(true_labels, labels))

    return {
        "pairwise": pairwise,
        "AMI": ami,
        "v_measure": v_measure,
        "homogeneity": homogeneity,
        "completeness": completeness,
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
    log.info(
        f"  V-measure={metrics['v_measure']:.4f}  "
        f"Homogeneity={metrics['homogeneity']:.4f}  "
        f"Completeness={metrics['completeness']:.4f}"
    )

    # ── Orthogroup type metrics (always-on) ──
    try:
        true_labels_arr = extract_true_labels(prot_meta)
        species_labels = extract_species_labels(prot_meta)
        og_types = compute_orthogroup_type_metrics(
            true_labels_arr, labels, species_labels, logger=log
        )
        metrics["orthogroup_types"] = og_types
    except Exception as e:
        log.debug(f"Orthogroup type metrics failed: {e}")

    # ── Per-species singleton diagnostics (always-on) ──
    try:
        sp_diag = compute_per_species_diagnostics(labels, prot_meta, logger=log)
        metrics["species_diagnostics"] = sp_diag
    except Exception as e:
        log.debug(f"Per-species diagnostics failed: {e}")

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

