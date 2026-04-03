import logging
import numpy as np
from collections import Counter
from .evaluation import relabel_contiguous

PYGEN_SWEEP_KWARGS = {
    # Keep PyGen focused on a smaller, biologically useful span and avoid
    # expensive outputs that this pipeline does not consume.
    "min_scale": -3.0,
    "max_scale": -0.5,
    "n_scale": 12,
    "n_tries": 10,
    "with_NVI": False,
    "with_ttprime": False,
    "with_optimal_scales": False,
    "tqdm_disable": True,
}

PROFILE_RESOLUTION_RANGE = (0.01, 1.0)
PROFILE_MIN_DIFF_RESOLUTION = 0.05


# ---------------------------------------------------------------------------
#  Helper: run PyGen on a single adjacency sub-matrix
# ---------------------------------------------------------------------------

def _pygen_sweep_on_adjacency(adj_sub, n_sub, sweep_kwargs, logger):
    """Run the PyGenStability Markov-stability sweep on *adj_sub* and return
    the raw results dict.  Row-normalizes the adjacency first."""
    from pygenstability import run

    log = logger
    row_sums = np.array(adj_sub.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj_sub.multiply(1.0 / row_sums[:, None])
    return run(adj_norm, **sweep_kwargs)


def _extract_pygen_levels(results, mask, N, logger):
    """Extract feasible, deduplicated levels from a PyGen results dict and map
    the subset labels back to the full *N*-node label space via *mask*."""
    log = logger
    levels = []

    if not (isinstance(results, dict) and "community_id" in results and "scales" in results):
        log.error("Unknown PyGenStability result format (expected dict with 'scales' and 'community_id')")
        return levels

    scales = results["scales"]
    community_id = results["community_id"]
    indices = np.arange(len(scales))

    last_n_clusters = -1
    last_labels = None

    for i in indices:
        scale_val = scales[i]
        sub_labels = np.array(community_id[i], dtype=np.int32)

        full_labels = np.full(N, -1, dtype=np.int32)
        full_labels[mask] = sub_labels

        full_labels, _ = relabel_contiguous(full_labels)
        n_clusters = len(np.unique(full_labels[full_labels >= 0]))

        if n_clusters == last_n_clusters and last_labels is not None and np.array_equal(full_labels, last_labels):
            continue

        last_n_clusters = n_clusters
        last_labels = full_labels.copy()

        n_mask = int(mask.sum())
        if n_clusters > 1 and n_clusters < 0.8 * n_mask:
            levels.append({
                "resolution": float(scale_val),
                "n_clusters": n_clusters,
                "labels": full_labels,
                "sizes": Counter(full_labels[full_labels >= 0].tolist()),
            })

    return levels


# ---------------------------------------------------------------------------
#  Public: PyGenStability discovery (per-component)
# ---------------------------------------------------------------------------

def run_pygenstability_discovery(
    snn_graph_result,
    *,
    per_component=True,
    min_component_size=50,
    logger=None,
):
    """Run PyGenStability to find robust topological partitions as candidate
    scales.

    When *per_component* is True (default), the sweep runs independently on
    every connected component with at least *min_component_size* nodes.
    Smaller components are assigned their own cluster label at every scale.
    """
    log = logger or logging.getLogger("multiscale")
    try:
        from pygenstability import run as _unused_import_check  # noqa: F401
    except ImportError:
        log.error("PyGenStability is not installed.  pip install pygenstability")
        raise

    adj = snn_graph_result["adjacency"].astype(np.float64)
    N = snn_graph_result["n_nodes"]

    log.info("Step 3 (PyGenStability): Discovering candidate stable regions via Markov Time sweeps...")
    log.info(
        "  PyGen sweep config: "
        f"min_scale={PYGEN_SWEEP_KWARGS['min_scale']}, "
        f"max_scale={PYGEN_SWEEP_KWARGS['max_scale']}, "
        f"n_scale={PYGEN_SWEEP_KWARGS['n_scale']}, "
        f"n_tries={PYGEN_SWEEP_KWARGS['n_tries']}"
    )

    from scipy.sparse.csgraph import connected_components
    n_comp, comp_labels = connected_components(adj, directed=False)

    # ------------------------------------------------------------------
    # Per-component discovery
    # ------------------------------------------------------------------
    if per_component and n_comp > 1:
        comp_sizes = np.bincount(comp_labels)
        sizable = [(cid, sz) for cid, sz in enumerate(comp_sizes) if sz >= min_component_size]
        small_ids = [cid for cid, sz in enumerate(comp_sizes) if sz < min_component_size]

        log.info(
            f"  Graph has {n_comp} components. "
            f"{len(sizable)} sizable (>={min_component_size}), "
            f"{len(small_ids)} small (assigned as singleton clusters)."
        )

        # Collect levels from each sizable component separately
        all_component_levels = []  # list of level-lists
        global_label_offset = 0

        for comp_id, comp_sz in sorted(sizable, key=lambda x: -x[1]):
            mask = comp_labels == comp_id
            adj_sub = adj[mask][:, mask].copy()

            log.info(f"  ▸ Component {comp_id}: {comp_sz} nodes — running PyGen sweep")
            results = _pygen_sweep_on_adjacency(adj_sub, comp_sz, PYGEN_SWEEP_KWARGS, log)
            comp_levels = _extract_pygen_levels(results, mask, N, log)

            # Offset labels so they don't collide with other components
            for lvl in comp_levels:
                labels = lvl["labels"]
                valid_mask = labels >= 0
                if valid_mask.any():
                    labels[valid_mask] += global_label_offset
                    max_label = int(labels[valid_mask].max())
                    global_label_offset = max_label + 1
                lvl["labels"] = labels
                lvl["sizes"] = Counter(labels[labels >= 0].tolist())

            all_component_levels.append(comp_levels)
            log.info(f"    → {len(comp_levels)} feasible levels from component {comp_id}")

        # Assign small components as singleton clusters at every scale
        # We need to figure out how many scale-slots we have
        n_scales_max = max((len(cl) for cl in all_component_levels), default=0)

        # Merge levels across components: align by index (coarse-to-fine order)
        if n_scales_max == 0:
            log.warning("  ⚠ No sizable component produced feasible PyGen levels.")
            levels = []
        else:
            # Standardize: pad shorter component level lists to n_scales_max
            for cl in all_component_levels:
                while len(cl) < n_scales_max:
                    if cl:
                        cl.append({
                            "resolution": cl[-1]["resolution"],
                            "n_clusters": cl[-1]["n_clusters"],
                            "labels": cl[-1]["labels"].copy(),
                            "sizes": dict(cl[-1]["sizes"]),
                        })
                    # else: will be handled by the fallback below

            # Build merged levels
            levels = []
            for scale_idx in range(n_scales_max):
                merged_labels = np.full(N, -1, dtype=np.int32)
                total_k = 0
                res_vals = []

                for cl in all_component_levels:
                    if scale_idx < len(cl):
                        lvl = cl[scale_idx]
                        valid = lvl["labels"] >= 0
                        merged_labels[valid] = lvl["labels"][valid]
                        total_k += lvl["n_clusters"]
                        res_vals.append(lvl["resolution"])

                # Assign small-component nodes their own cluster ids
                for small_cid in small_ids:
                    small_mask = comp_labels == small_cid
                    small_nodes = np.where(small_mask)[0]
                    for node in small_nodes:
                        merged_labels[node] = global_label_offset
                        global_label_offset += 1
                        total_k += 1

                merged_labels, _ = relabel_contiguous(merged_labels)
                n_clusters = len(np.unique(merged_labels[merged_labels >= 0]))

                levels.append({
                    "resolution": float(np.mean(res_vals)) if res_vals else 0.0,
                    "n_clusters": n_clusters,
                    "labels": merged_labels,
                    "sizes": Counter(merged_labels[merged_labels >= 0].tolist()),
                })

    else:
        # ------------------------------------------------------------------
        # Single-component or per_component=False: original LCC-only path
        # ------------------------------------------------------------------
        if n_comp > 1:
            log.warning(f"  ⚠ Graph has {n_comp} disconnected components (per_component disabled).")
            log.info("  Extracting the largest connected component for Markov stability sweep...")
            largest_comp_id = np.argmax(np.bincount(comp_labels))
            mask = comp_labels == largest_comp_id
        else:
            mask = np.ones(N, dtype=bool)

        adj_sub = adj[mask][:, mask].copy()
        results = _pygen_sweep_on_adjacency(adj_sub, int(mask.sum()), PYGEN_SWEEP_KWARGS, log)
        levels = _extract_pygen_levels(results, mask, N, log)

    # ------------------------------------------------------------------
    # Common post-processing
    # ------------------------------------------------------------------
    levels = sorted(levels, key=lambda x: x["n_clusters"])
    for i, lvl in enumerate(levels):
        log.info(f"  Level {i}: K={lvl['n_clusters']} (Resolution: {lvl['resolution']:.4f})")

    if not levels:
        log.warning("  ⚠ PyGen stability returned no feasible scales.")

    return {"levels": levels}


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
        log.info(f"  Per-component profile: {len(sizable)} sizable components")

        global_label_offset = 0
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

            for part in profile:
                sub_labels = np.array(part.membership, dtype=np.int32)
                full_labels = np.full(N, -1, dtype=np.int32)
                full_labels[mask] = sub_labels

                full_labels, _ = relabel_contiguous(full_labels)
                valid = full_labels >= 0
                if valid.any():
                    full_labels[valid] += global_label_offset
                    global_label_offset = int(full_labels[valid].max()) + 1

                n_clusters = len(np.unique(full_labels[full_labels >= 0]))
                res = getattr(part, 'resolution_parameter', 0.0)

                if n_clusters > 1 and n_clusters < 0.8 * comp_sz:
                    levels.append({
                        "resolution": res,
                        "n_clusters": n_clusters,
                        "labels": full_labels,
                        "sizes": Counter(full_labels[full_labels >= 0].tolist()),
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
