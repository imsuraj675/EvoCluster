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

def _compute_mean_conductance(labels, g, cluster_ids):
    """Compute size-weighted mean conductance across clusters."""
    total_vol = 2 * g.ecount()
    if total_vol == 0:
        return 0.0

    conductances = []
    sizes = []
    for cid in cluster_ids:
        members = np.where(labels == cid)[0]
        n_members = len(members)
        if n_members < 1:
            continue

        subgraph = g.subgraph(members.tolist())
        internal_edges = subgraph.ecount()
        vol_c = sum(g.degree(m) for m in members.tolist())
        cut_c = vol_c - 2 * internal_edges
        vol_complement = total_vol - vol_c
        denom = min(vol_c, vol_complement)

        if denom > 0:
            conductances.append(cut_c / denom)
        else:
            conductances.append(0.0)
        sizes.append(n_members)

    if not conductances:
        return 0.0

    weights = np.array(sizes, dtype=np.float64)
    mean_cond = float(np.average(conductances, weights=weights))
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
#  Elbow detection (simple second-derivative / kneedle-lite)
# ---------------------------------------------------------------------------

def _find_elbow(values):
    """Return the index of the elbow point in a sorted-descending score array.

    Uses a simplified kneedle approach: normalise the curve to [0,1] on both
    axes, compute the perpendicular distance of each point from the line
    connecting the first and last points, and return the index of the maximum
    distance.
    """
    n = len(values)
    if n <= 2:
        return 0

    x = np.linspace(0.0, 1.0, n)
    y_min, y_max = float(np.min(values)), float(np.max(values))
    if y_max - y_min < 1e-12:
        return 0
    y = (np.array(values, dtype=np.float64) - y_min) / (y_max - y_min)

    # Line from first to last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return 0
    line_unit = line_vec / line_len

    dists = np.zeros(n)
    for i in range(n):
        pt = np.array([x[i], y[i]]) - p1
        proj = np.dot(pt, line_unit)
        closest = p1 + proj * line_unit
        dists[i] = np.linalg.norm(np.array([x[i], y[i]]) - closest)

    return int(np.argmax(dists))


# ---------------------------------------------------------------------------
#  Main scorer & selector
# ---------------------------------------------------------------------------

def score_and_select_scales(
    hierarchy,
    multiscale_result,
    snn_graph_result,
    *,
    selection_policy="best_composite",
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
    selection_policy : str
        ``"best_composite"`` — default composite scorer with target-aware
        banding.
        ``"elbow"`` — retain all levels above the elbow in the composite
        score curve.
        ``"max_stability"`` — pick the top N levels by consensus-stability
        alone.
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

    log.info(f"Step 4: Scoring stability across {n_levels} levels (policy={selection_policy})")
    target_K = max(N * target_cluster_ratio, 10.0)
    target_avg_size = N / target_K
    singleton_tolerance = 0.10

    per_level_scores = []
    feasible_indices = []

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
        densities = []
        for cid in cluster_ids:
            members = np.where(labels == cid)[0]
            n_members = len(members)
            if n_members < 2:
                densities.append(0.0)
                continue
            possible_edges = n_members * (n_members - 1) / 2
            subgraph = g.subgraph(members.tolist())
            actual_edges = subgraph.ecount()
            densities.append(actual_edges / possible_edges if possible_edges > 0 else 0.0)

        if cluster_ids.size > 0:
            sizes_arr = np.array([np.sum(labels == cid) for cid in cluster_ids], dtype=np.float64)
            cohesion = float(np.average(densities, weights=sizes_arr))
        else:
            cohesion = 0.0

        if feasible and n_clusters > 1:
            separation = _compute_mean_conductance(labels, g, cluster_ids)
        else:
            separation = 0.0

        avg_cluster_size = N / max(n_clusters, 1)
        size_reg = min(avg_cluster_size / target_avg_size, 1.0)
        n_size1_clusters = sum(1 for cid in cluster_ids if np.sum(labels == cid) == 1)
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

    if selection_policy == "elbow":
        selected = _select_elbow(
            feasible_indices, composites, levels, max_selected_levels, log
        )
    elif selection_policy == "max_stability":
        selected = _select_max_stability(
            feasible_indices, per_level_scores, levels, max_selected_levels, log
        )
    else:
        # best_composite (default)
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
        "policy_used": selection_policy,
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

    # Define the biological bounds where an anchor is mathematically sane
    plausible_indices = [
        i for i in feasible_indices 
        if (N / 15) <= levels[i]["n_clusters"] <= (N / 2)
    ]
    
    if plausible_indices:
        # Trust the composite score (which now includes the target penalty)
        target_idx = max(plausible_indices, key=lambda i: composites[i])
    else:
        # Extreme fallback
        target_idx = max(feasible_indices, key=lambda i: composites[i])

    selected.append(target_idx)
    selection_trace.append(
        ("target_anchor", levels[target_idx]["resolution"], levels[target_idx]["n_clusters"])
    )

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


def _select_elbow(feasible_indices, composites, levels, max_selected_levels, log):
    """Select all feasible levels above the elbow in the composite curve."""
    if not feasible_indices:
        return []

    feas_composites = np.array([composites[i] for i in feasible_indices])
    sorted_order = np.argsort(-feas_composites)  # descending
    sorted_scores = feas_composites[sorted_order]

    elbow_pos = _find_elbow(sorted_scores)
    # Keep everything at or above the elbow
    n_keep = min(elbow_pos + 1, max_selected_levels)
    n_keep = max(n_keep, 1)

    selected_feas = sorted_order[:n_keep]
    selected = [feasible_indices[i] for i in selected_feas]

    log.info(
        f"  Elbow policy: elbow at rank {elbow_pos} "
        f"(composite={sorted_scores[elbow_pos]:.3f}), keeping {n_keep} levels"
    )
    return selected


def _select_max_stability(feasible_indices, per_level_scores, levels,
                          max_selected_levels, log):
    """Select the top N levels by consensus stability alone."""
    if not feasible_indices:
        return []

    ranked = sorted(
        feasible_indices,
        key=lambda i: per_level_scores[i]["consensus"],
        reverse=True,
    )
    selected = ranked[:max_selected_levels]

    log.info(
        f"  max_stability policy: top {len(selected)} by consensus NMI — "
        + str([
            {"idx": i, "K": levels[i]["n_clusters"],
             "consensus": round(per_level_scores[i]["consensus"], 3)}
            for i in selected
        ])
    )
    return selected

