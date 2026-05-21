import logging
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

DEFAULT_TARGET_CLUSTER_RATIO = 1.0 / 3.0
DEFAULT_MAX_SELECTED_LEVELS = 6
K_SPACING_FRACTION = 0.10

def _is_feasible(n_clusters, n_singletons, N, logger=None):
    """Check if a Leiden level is feasible (not pathological)."""
    avg_size = N / max(n_clusters, 1)
    singleton_ratio = n_singletons / max(N, 1)

    if n_clusters < 2:
        return False, f"K={n_clusters} < 2 (trivial single cluster)"
    if avg_size < 1.3:
        return False, f"avg_size={avg_size:.1f} < 1.3 (near-singleton)"
    if singleton_ratio > 0.5:
        return False, f"singleton_ratio={singleton_ratio:.2f} > 0.50"
    if n_clusters > 0.8 * N:
        return False, f"K={n_clusters} > 0.8*N={int(0.8*N)} (trivial partition)"
    return True, "ok"

def _compute_mean_conductance_fast(labels, cluster_ids, sizes_arr, internal_counts,
                                    degrees, total_vol):
    """Compute size-weighted mean conductance without per-cluster subgraph extraction.

    Uses pre-computed internal edge counts and degree array so the cost is
    O(N) instead of O(K * (N + E)).
    """
    if total_vol == 0 or len(cluster_ids) == 0:
        return 0.0

    max_cid = int(cluster_ids.max())

    # Volume per cluster: sum of degrees of members — O(N)
    valid_mask = labels >= 0
    vol_per_cluster = np.zeros(max_cid + 1, dtype=np.float64)
    np.add.at(vol_per_cluster, labels[valid_mask], degrees[valid_mask].astype(np.float64))

    vol_c = vol_per_cluster[cluster_ids]
    cut_c = vol_c - 2.0 * internal_counts[cluster_ids].astype(np.float64)
    vol_complement = total_vol - vol_c
    denom = np.minimum(vol_c, vol_complement)

    cond_values = np.zeros_like(cut_c, dtype=np.float64)
    np.divide(cut_c, denom, out=cond_values, where=denom > 0)

    # Filter to clusters with >= 1 member (matching original skip logic)
    valid = sizes_arr >= 1
    if not valid.any():
        return 0.0

    mean_cond = float(np.average(cond_values[valid], weights=sizes_arr[valid]))
    mean_cond = max(0.0, min(1.0, mean_cond))
    return 1.0 - mean_cond

def _consensus_nmi(g, resolution, objective_function="CPM", n_runs=5, base_seed=0):
    """Run leidenalg n_runs times with different seeds, return mean pairwise NMI."""
    import leidenalg as la
    partitions = []
    weights = g.es['weight'] if 'weight' in g.edge_attributes() else None

    if objective_function.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition

    for i in range(n_runs):
        part = la.find_partition(
            g,
            partition_type,
            weights=weights,
            resolution_parameter=resolution,
            n_iterations=-1,
            seed=base_seed + i,
        )
        partitions.append(np.array(part.membership, dtype=np.int32))

    if len(partitions) < 2:
        return 1.0

    nmis = []
    for a_idx in range(len(partitions)):
        for b_idx in range(a_idx + 1, len(partitions)):
            nmis.append(float(normalized_mutual_info_score(
                partitions[a_idx], partitions[b_idx]
            )))
    return float(np.mean(nmis)) if nmis else 1.0


# ---------------------------------------------------------------------------
#  Main scorer & selector
# ---------------------------------------------------------------------------

def score_and_select_scales(
    hierarchy,
    multiscale_result,
    snn_graph_result,
    *,
    consensus_runs=5,
    seed=0,
    objective_function="CPM",
    target_cluster_ratio=None,
    max_selected_levels=None,
    logger=None,
):
    """Score levels and select a target-aware coarse-to-fine subset.

    Parameters
    ----------
    target_cluster_ratio : float or None
        Soft prior for where the biological optimum K lives relative to N.
        Default ``1/3``.  Used as a Gaussian penalty centre, not a hard gate.
    max_selected_levels : int or None
        Maximum number of levels to retain.  Default ``6``.
    """
    log = logger or logging.getLogger("multiscale")

    if target_cluster_ratio is None:
        target_cluster_ratio = DEFAULT_TARGET_CLUSTER_RATIO
    if max_selected_levels is None:
        max_selected_levels = DEFAULT_MAX_SELECTED_LEVELS

    levels = multiscale_result["levels"]
    n_levels = len(levels)
    g = snn_graph_result["graph"]
    N = snn_graph_result["n_nodes"]

    log.info(f"Step 4: Scoring stability across {n_levels} levels")
    target_K = max(N * target_cluster_ratio, 10.0)
    target_avg_size = N / target_K
    singleton_tolerance = 0.10

    per_level_scores = []
    feasible_indices = []

    # Pre-compute edge endpoints and degrees for vectorized graph metrics
    _edge_list = g.get_edgelist()
    if _edge_list:
        _edge_arr = np.array(_edge_list, dtype=np.int64)
        _edge_sources = _edge_arr[:, 0]
        _edge_targets = _edge_arr[:, 1]
    else:
        _edge_sources = np.array([], dtype=np.int64)
        _edge_targets = np.array([], dtype=np.int64)
    _degrees = np.array(g.degree(), dtype=np.int64)
    _total_vol = 2 * g.ecount()

    for lev_idx in range(n_levels):
        labels = levels[lev_idx]["labels"]
        res = levels[lev_idx]["resolution"]
        n_clusters = levels[lev_idx]["n_clusters"]
        n_singletons = int(np.sum(labels == -1))
        feasible, reason = _is_feasible(n_clusters, n_singletons, N, logger=log)

        nmi = 1.0
        if lev_idx > 0:
            prev_labels = levels[lev_idx - 1]["labels"]
            mask = (labels >= 0) & (prev_labels >= 0)
            if mask.sum() > 0:
                nmi = float(normalized_mutual_info_score(prev_labels[mask], labels[mask]))

        prev_k = levels[lev_idx - 1]["n_clusters"] if lev_idx > 0 else max(n_clusters, 1)
        count_ratio = n_clusters / max(prev_k, 1)

        cluster_ids = np.unique(labels[labels >= 0])

        if cluster_ids.size > 0:
            max_cid = int(cluster_ids.max())
            _valid = labels >= 0

            # Cluster sizes via bincount — O(N) instead of O(K*N)
            sizes_full = np.bincount(labels[_valid], minlength=max_cid + 1)
            sizes_arr = sizes_full[cluster_ids].astype(np.float64)

            # Internal edges per cluster via single edge walk — O(E) instead of K subgraphs
            if _edge_sources.size > 0:
                _cs = labels[_edge_sources]
                _ct = labels[_edge_targets]
                _int_mask = (_cs >= 0) & (_cs == _ct)
                if _int_mask.any():
                    internal_counts = np.bincount(_cs[_int_mask], minlength=max_cid + 1)
                else:
                    internal_counts = np.zeros(max_cid + 1, dtype=np.int64)
            else:
                internal_counts = np.zeros(max_cid + 1, dtype=np.int64)

            # Cohesion: vectorized density = internal_edges / possible_edges
            n_members_arr = sizes_full[cluster_ids]
            possible_arr = n_members_arr * (n_members_arr - 1) / 2.0
            internal_arr = internal_counts[cluster_ids].astype(np.float64)
            densities_arr = np.zeros(len(cluster_ids), dtype=np.float64)
            np.divide(internal_arr, possible_arr, out=densities_arr, where=possible_arr > 0)
            cohesion = float(np.average(densities_arr, weights=sizes_arr))

            n_size1_clusters = int(np.sum(sizes_full[cluster_ids] == 1))
        else:
            cohesion = 0.0
            sizes_arr = np.array([], dtype=np.float64)
            n_size1_clusters = 0
            internal_counts = np.zeros(0, dtype=np.int64)

        if feasible and n_clusters > 1:
            separation = _compute_mean_conductance_fast(
                labels, cluster_ids, sizes_arr, internal_counts,
                _degrees, _total_vol
            )
        else:
            separation = 0.0

        avg_cluster_size = N / max(n_clusters, 1)
        size_reg = min(avg_cluster_size / target_avg_size, 1.0)
        effective_singleton_ratio = (n_singletons + n_size1_clusters) / max(N, 1)
        frag_penalty = max(0.0, effective_singleton_ratio - singleton_tolerance)

        consensus = 1.0
        if feasible and consensus_runs > 1:
            consensus = _consensus_nmi(
                g, res, objective_function=objective_function,
                n_runs=consensus_runs, base_seed=seed,
            )

        # ── Soft target prior (Gaussian penalty) ──
        # Instead of hard-coding band anchoring, add a smooth bonus that
        # peaks at target_K and decays as K moves away.
        if feasible and n_clusters > 0:
            target_penalty = float(np.exp(
                -0.5 * ((n_clusters - target_K) / (0.3 * target_K + 1e-6)) ** 2
            ))
        else:
            target_penalty = 0.0

        if feasible:
            composite = (
                0.18 * nmi
                + 0.18 * cohesion
                + 0.18 * separation
                + 0.12 * size_reg
                + 0.10 * consensus
                + 0.10 * target_penalty
                - 0.14 * frag_penalty
            )
        else:
            composite = -1.0

        size_drift = min(count_ratio / 3.0, 1.0) if count_ratio > 1.0 else 0.0

        # ── Dominant factor tracking ──
        factor_contribs = {
            "nmi": 0.18 * nmi,
            "cohesion": 0.18 * cohesion,
            "separation": 0.18 * separation,
            "size_reg": 0.12 * size_reg,
            "consensus": 0.10 * consensus,
            "target_prior": 0.10 * target_penalty,
            "frag_penalty": -0.14 * frag_penalty,
        }
        dominant_factor = max(factor_contribs, key=lambda k: factor_contribs[k]) if feasible else "infeasible"

        per_level_scores.append({
            "resolution": res, "nmi": nmi, "count_ratio": count_ratio,
            "cohesion": cohesion, "separation": separation, "size_regularizer": size_reg,
            "fragmentation_penalty": frag_penalty, "consensus": consensus, "size_drift": size_drift,
            "target_penalty": target_penalty, "dominant_factor": dominant_factor,
            "composite": composite, "n_clusters": n_clusters, "feasible": feasible, "feasible_reason": reason,
        })
        if feasible:
            feasible_indices.append(lev_idx)

        feas_tag = "✓" if feasible else "✗"
        log.info(
            f"  {feas_tag} res={res:.4f}  K={n_clusters:>5d}  "
            f"NMI={nmi:.3f}  coh={cohesion:.3f}  sep={separation:.3f}  "
            f"szreg={size_reg:.3f}  frag={frag_penalty:.3f}  cons={consensus:.3f}  "
            f"tgt={target_penalty:.3f}  comp={composite:.3f}  [{dominant_factor}]"
        )
        if not feasible:
            log.info(f"    → INFEASIBLE: {reason}")
        if cohesion < 0.05 and n_clusters > 5 and feasible:
            log.warning(f"  ⚠ res={res:.4f}: very low cohesion ({cohesion:.3f}) — clusters are internally disconnected!")
        if nmi < 0.3 and lev_idx > 0:
            log.warning(f"  ⚠ res={res:.4f}: NMI={nmi:.3f} vs previous level — big structural shift!")

    composites = [s["composite"] for s in per_level_scores]

    # ── Plateau bonus ──
    if len(feasible_indices) >= 3:
        feas_ks = [levels[i]["n_clusters"] for i in feasible_indices]
        log_ks = np.log1p(feas_ks)
        gradients = np.abs(np.diff(log_ks))
        median_grad = float(np.median(gradients)) if len(gradients) > 0 else 1.0
        plateau_threshold = 0.5 * median_grad if median_grad > 0 else 0.1
        plateau_scores = np.zeros(len(feasible_indices))
        for i in range(len(feasible_indices)):
            left_ok = (i == 0) or (gradients[i - 1] < plateau_threshold)
            right_ok = (i == len(feasible_indices) - 1) or (gradients[i] < plateau_threshold)
            if left_ok and right_ok:
                plateau_scores[i] = 1.0
            elif left_ok or right_ok:
                plateau_scores[i] = 0.5

        for idx, feas_idx in enumerate(feasible_indices):
            bonus = 0.10 * plateau_scores[idx]
            composites[feas_idx] += bonus
            per_level_scores[feas_idx]["plateau_bonus"] = bonus
            per_level_scores[feas_idx]["composite"] = composites[feas_idx]
            if plateau_scores[idx] > 0:
                log.debug(f"  Plateau bonus +{bonus:.3f} for res={levels[feas_idx]['resolution']:.4f}")

    # ==================================================================
    #  Policy-based selection
    # ==================================================================

    selected = _select_best_composite(
        feasible_indices, composites, levels, N, target_K,
        max_selected_levels, log
    )

    selected = sorted(set(selected), key=lambda i: levels[i]["n_clusters"])
    if not selected:
        log.warning("  ⚠ No feasible levels found! Falling back to all levels.")
        selected = list(range(n_levels))
    k_values = [per_level_scores[i]["n_clusters"] for i in selected]
    log.info(f"  ▸ Selected {len(selected)} level(s): {selected} (K: {k_values})")
    return {
        "per_level_scores": per_level_scores,
        "selected_levels": selected,
        "feasible_indices": feasible_indices,
    }


# ---------------------------------------------------------------------------
#  Selection policies
# ---------------------------------------------------------------------------

def _select_best_composite(feasible_indices, composites, levels, N, target_K,
                           max_selected_levels, log):
    """Target-aware banding with diversity fill (original policy, soft prior)."""
    if len(feasible_indices) <= max_selected_levels:
        if feasible_indices:
            log.info(
                f"  Retaining all {len(feasible_indices)} feasible level(s) "
                f"(<= selection cap {max_selected_levels})."
            )
        return list(feasible_indices)
    
    selected = []
    selection_trace = []

    bands = [
        ("macro",        lambda K: K < N / 20),
        ("ultra_coarse", lambda K: N / 20 <= K < N / 10),
        ("coarse",       lambda K: N / 10 <= K < N / 6),
        ("mid_coarse",   lambda K: N / 6  <= K < N / 4),
        ("target",       lambda K: N / 4  <= K < N / 2),
        ("fine",         lambda K: N / 2  <= K < 0.7 * N),
    ]

    target_idx = min(
        feasible_indices,
        key=lambda i: (
            abs(levels[i]["n_clusters"] - target_K),
            -composites[i],
        ),
    )
    selected.append(target_idx)
    selection_trace.append(
        ("target_anchor", levels[target_idx]["resolution"], levels[target_idx]["n_clusters"])
    )
    log.info(
        f"  Target-aware anchor: target_K≈{target_K:.0f}, "
        f"selected res={levels[target_idx]['resolution']:.4f} "
        f"(K={levels[target_idx]['n_clusters']})"
    )

    for band_name, band_filter in bands:
        candidates = [i for i in feasible_indices if band_filter(levels[i]["n_clusters"]) and i not in selected]
        if candidates:
            best = max(candidates, key=lambda i: composites[i])
            selected.append(best)
            selection_trace.append(
                (band_name, levels[best]["resolution"], levels[best]["n_clusters"])
            )

    remaining = sorted(
        [i for i in feasible_indices if i not in selected],
        key=lambda i: composites[i], reverse=True,
    )

    while len(selected) < max_selected_levels and remaining:
        cand_idx = remaining.pop(0)
        cand_k = levels[cand_idx]["n_clusters"]

        is_diverse = True
        for sel_idx in selected:
            sel_k = levels[sel_idx]["n_clusters"]
            if abs(cand_k - sel_k) / max(sel_k, 1) < K_SPACING_FRACTION:
                is_diverse = False
                break

        if is_diverse:
            selected.append(cand_idx)
            selection_trace.append(
                ("diversity_fill", levels[cand_idx]["resolution"], levels[cand_idx]["n_clusters"])
            )

    log.info(
        "  Target-aware selection trace: "
        + str([
            {"source": source, "resolution": round(resolution, 4), "K": n_clusters}
            for source, resolution, n_clusters in selection_trace
        ])
    )
    return selected
