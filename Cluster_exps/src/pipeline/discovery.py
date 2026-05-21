import logging
import numpy as np
from collections import Counter
from .evaluation import relabel_contiguous

PROFILE_RESOLUTION_RANGE = (0.01, 1.0)
PROFILE_MIN_DIFF_RESOLUTION = 0.05


# ---------------------------------------------------------------------------
#  Public: Resolution profile discovery (per-component)
# ---------------------------------------------------------------------------

def run_resolution_profile_discovery(
    snn_graph_result,
    resolution_range=PROFILE_RESOLUTION_RANGE,
    objective_function="CPM",
    per_component=True,
    min_component_size=50,
    logger=None,
):
    """Run leidenalg Resolution Profile to find structural graph transitions.

    When *per_component* is True, profiles are computed on each sizable
    connected component independently (same logic as PyGen per-component).
    """
    log = logger or logging.getLogger("multiscale")
    try:
        import leidenalg as la
    except ImportError:
        log.error("leidenalg is not installed.  pip install leidenalg")
        raise

    g = snn_graph_result["graph"]
    N = snn_graph_result["n_nodes"]
    weights = g.es['weight'] if 'weight' in g.edge_attributes() else None

    if objective_function.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition

    log.info(f"Step 3 (Resolution Profile): Scanning discrete graph transitions across resolution span {resolution_range}...")

    # ------------------------------------------------------------------
    # Run profile (optionally per-component)
    # ------------------------------------------------------------------
    from scipy.sparse.csgraph import connected_components
    adj = snn_graph_result["adjacency"]
    n_comp, comp_labels = connected_components(adj, directed=False)

    levels = []

    if per_component and n_comp > 1:
        comp_sizes = np.bincount(comp_labels)
        sizable = [(cid, sz) for cid, sz in enumerate(comp_sizes) if sz >= min_component_size]
        small_ids = [cid for cid, sz in enumerate(comp_sizes) if sz < min_component_size]
        singleton_component_ids = list(small_ids)
        log.info(
            f"  Per-component profile: {len(sizable)} sizable components, "
            f"{len(small_ids)} small components assigned as singleton clusters"
        )

        component_level_sets = []
        for comp_id, comp_sz in sorted(sizable, key=lambda x: -x[1]):
            mask = comp_labels == comp_id
            members = np.where(mask)[0].tolist()
            sub_g = g.subgraph(members)
            sub_weights = sub_g.es['weight'] if 'weight' in sub_g.edge_attributes() else None

            optimiser = la.Optimiser()
            profile = optimiser.resolution_profile(
                sub_g, partition_type,
                resolution_range=resolution_range,
                weights=sub_weights,
                min_diff_resolution=PROFILE_MIN_DIFF_RESOLUTION,
            )
            profile = sorted(profile, key=lambda p: getattr(p, 'resolution_parameter', 0.0))

            comp_levels = []
            for part in profile:
                sub_labels = np.array(part.membership, dtype=np.int32)
                sub_labels, _ = relabel_contiguous(sub_labels)
                n_clusters = len(np.unique(sub_labels[sub_labels >= 0]))
                res = getattr(part, 'resolution_parameter', 0.0)

                if n_clusters > 1 and n_clusters < 0.8 * comp_sz:
                    comp_levels.append({
                        "resolution": res,
                        "n_clusters": n_clusters,
                        "labels": sub_labels,
                        "mask": mask,
                        "sizes": Counter(sub_labels[sub_labels >= 0].tolist()),
                    })

            comp_levels = sorted(comp_levels, key=lambda x: x["n_clusters"])
            component_level_sets.append(comp_levels)
            if not comp_levels:
                singleton_component_ids.append(comp_id)
            log.info(f"    Component {comp_id}: {len(comp_levels)} feasible profile levels")

        n_scales_max = max((len(cl) for cl in component_level_sets), default=0)

        if n_scales_max == 0:
            log.warning("  ⚠ No sizable component produced feasible resolution-profile levels.")
            if singleton_component_ids:
                singleton_labels = np.full(N, -1, dtype=np.int32)
                next_label = 0
                for singleton_cid in singleton_component_ids:
                    for node in np.where(comp_labels == singleton_cid)[0]:
                        singleton_labels[node] = next_label
                        next_label += 1
                singleton_labels, _ = relabel_contiguous(singleton_labels)
                levels = [{
                    "resolution": 0.0,
                    "n_clusters": len(np.unique(singleton_labels[singleton_labels >= 0])),
                    "labels": singleton_labels,
                    "sizes": Counter(singleton_labels[singleton_labels >= 0].tolist()),
                }]
            else:
                levels = []
        else:
            # Align component-local ladders by coarse-to-fine index. If a
            # component yields fewer profile levels, keep its finest available
            # partition for the remaining global scales so every returned level
            # covers the full dataset.
            for cl in component_level_sets:
                while len(cl) < n_scales_max and cl:
                    last = cl[-1]
                    cl.append({
                        "resolution": last["resolution"],
                        "n_clusters": last["n_clusters"],
                        "labels": last["labels"].copy(),
                        "mask": last["mask"],
                        "sizes": dict(last["sizes"]),
                    })

            for scale_idx in range(n_scales_max):
                merged_labels = np.full(N, -1, dtype=np.int32)
                next_label = 0
                res_vals = []

                for cl in component_level_sets:
                    if not cl or scale_idx >= len(cl):
                        continue
                    lvl = cl[scale_idx]
                    mask = lvl["mask"]
                    sub_labels = lvl["labels"]
                    valid_sub = sub_labels >= 0
                    if not valid_sub.any():
                        continue

                    sub_remap = {
                        old: next_label + offset
                        for offset, old in enumerate(sorted(np.unique(sub_labels[valid_sub]).tolist()))
                    }
                    component_full = np.full(mask.sum(), -1, dtype=np.int32)
                    for old, new in sub_remap.items():
                        component_full[sub_labels == old] = new
                    merged_labels[mask] = component_full
                    next_label += len(sub_remap)
                    res_vals.append(lvl["resolution"])

                # Preserve small disconnected components as explicit singleton
                # clusters rather than leaving them unassigned.
                for singleton_cid in singleton_component_ids:
                    small_nodes = np.where(comp_labels == singleton_cid)[0]
                    for node in small_nodes:
                        merged_labels[node] = next_label
                        next_label += 1

                merged_labels, _ = relabel_contiguous(merged_labels)
                n_clusters = len(np.unique(merged_labels[merged_labels >= 0]))

                levels.append({
                    "resolution": float(np.mean(res_vals)) if res_vals else 0.0,
                    "n_clusters": n_clusters,
                    "labels": merged_labels,
                    "sizes": Counter(merged_labels[merged_labels >= 0].tolist()),
                })
    else:
        # Single-component path (original)
        optimiser = la.Optimiser()
        profile = optimiser.resolution_profile(
            g, partition_type,
            resolution_range=resolution_range,
            weights=weights,
            min_diff_resolution=PROFILE_MIN_DIFF_RESOLUTION,
        )
        profile = sorted(profile, key=lambda p: getattr(p, 'resolution_parameter', 0.0))

        log.info("  Resolution profile curve before filtering:")
        for idx, part in enumerate(profile):
            labels = np.array(part.membership, dtype=np.int32)
            labels, _ = relabel_contiguous(labels)
            n_clusters = len(np.unique(labels[labels >= 0]))
            res = getattr(part, 'resolution_parameter', float(idx))

            log.info(f"    res={res:.4f} → K={n_clusters}")

            if n_clusters > 1 and n_clusters < 0.8 * N:
                levels.append({
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "labels": labels,
                    "sizes": Counter(labels[labels >= 0].tolist()),
                })

    # ------------------------------------------------------------------
    # K-spacing thinning (applied uniformly)
    # ------------------------------------------------------------------
    levels = sorted(levels, key=lambda x: x["n_clusters"])

    selected_levels = []
    if levels:
        selected_levels.append(levels[0])
        last_k = levels[0]["n_clusters"]

        for lvl in levels[1:]:
            step_threshold = max(2, int(last_k * 0.05))
            if lvl["n_clusters"] >= last_k + step_threshold:
                selected_levels.append(lvl)
                last_k = lvl["n_clusters"]

        if levels[-1]["resolution"] != selected_levels[-1]["resolution"]:
            if levels[-1]["n_clusters"] > selected_levels[-1]["n_clusters"]:
                selected_levels.append(levels[-1])

    selected_levels = sorted(selected_levels, key=lambda x: x["n_clusters"])

    log.info(f"  ▸ Profile thinned to {len(selected_levels)} candidate scales based on K-spacing.")
    for lvl in selected_levels:
        log.info(f"    Selected candidate: res={lvl['resolution']:.4f} → K={lvl['n_clusters']}")

    return {"levels": selected_levels}
