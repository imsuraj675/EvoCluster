import math
import heapq
import logging
from collections import defaultdict
import numpy as np

EPS = 1e-12

def refine_and_flatten(
    X,
    snn_graph_result,
    hierarchy,
    selected_scales,
    multiscale_result,
    *,
    centroid_cos_threshold=0.85,
    edge_connectivity_threshold=0.05,
    edge_strong_threshold=None,
    protect_singletons=True,
    output_level="adaptive",
    k_neighbors=None,
    homology_rescue=False,
    homology_rescue_cos=0.95,
    homology_rescue_max_size=20,
    cross_branch_rescue=False,
    cross_branch_cos=0.93,
    cross_branch_edge=0.03,
    max_cross_branch_merges=500,
    logger=None,
):
    from .evaluation import relabel_contiguous
    log = logger or logging.getLogger("multiscale")

    if edge_strong_threshold is None:
        edge_strong_threshold = min(edge_connectivity_threshold * 2.0, 0.30)

    levels = multiscale_result["levels"]
    selected = selected_scales["selected_levels"]
    adjacency = snn_graph_result["adjacency"]
    N = X.shape[0]

    k = k_neighbors or int(0.6 * math.sqrt(N))

    log.info("Step 5: Cascaded branch-local refinement")
    log.info(f"  Selected levels for cascade: {selected}")
    log.info(f"  Base thresholds — cos: {centroid_cos_threshold}, edge: {edge_connectivity_threshold}, strong: {edge_strong_threshold}")
    log.info(f"  Graph-aware normalization — k={k}")
    log.info(f"  Homology rescue: {homology_rescue} (cos>={homology_rescue_cos}, max_size<={homology_rescue_max_size})")
    log.info(f"  Cross-branch rescue: {cross_branch_rescue} (cos>={cross_branch_cos}, edge>={cross_branch_edge}, max={max_cross_branch_merges})")

    adj_csr = adjacency.tocsr() if not hasattr(adjacency, 'indptr') else adjacency

    n_stages = len(selected) - 1
    
    if len(selected) == 0:
        log.error("  CRITICAL: 0 levels selected for merge cascade! Returning all singletons.")
        single_labels = np.full(N, -1, dtype=np.int32)
        return {
            "labels": single_labels,
            "labels_all": {"coarse": single_labels, "fine": single_labels, "adaptive": single_labels},
            "n_clusters": 0,
            "n_singletons": N,
            "merge_log": [],
            "n_merges": 0,
            "stage_summaries": [],
        }
        
    if n_stages < 1:
        log.warning("  Only 1 level selected — skipping cascade merge.")
        single_labels = levels[selected[0]]["labels"].copy()
        single_out, _ = relabel_contiguous(single_labels)
        return {
            "labels": single_out,
            "labels_all": {"coarse": single_out, "fine": single_out, "adaptive": single_out},
            "n_clusters": len(set(single_out[single_out >= 0].tolist())),
            "n_singletons": int(np.sum(single_out == -1)),
            "merge_log": [],
            "n_merges": 0,
            "stage_summaries": [],
        }

    stage_thresholds = []
    for s_idx in range(n_stages):
        t_linear = s_idx / max(n_stages - 1, 1) if n_stages > 1 else 0.0
        # Quadratic escalation for modest early-stage loosening
        t_quad = t_linear ** 2
        
        cos_t  = centroid_cos_threshold    + 0.10 * t_quad
        edge_t = edge_connectivity_threshold + 0.05 * t_quad
        strong_t = edge_strong_threshold + 0.05 * t_quad
        stage_thresholds.append({
            "cos": min(cos_t, 0.98),
            "edge": min(edge_t, 0.50),
            "strong": min(strong_t, 0.50),
        })
        log.info(f"  Stage {s_idx+1}/{n_stages}: cos>={stage_thresholds[-1]['cos']:.3f} edge>={stage_thresholds[-1]['edge']:.3f} strong>={stage_thresholds[-1]['strong']:.3f}")

    total_merges = 0
    total_rejected = 0
    total_homology_rescues = 0
    all_merge_log = []
    merged_labels = {}
    stage_summaries = []

    for s_idx in range(n_stages):
        guide_idx = selected[s_idx]
        fine_idx = selected[s_idx + 1]
        thresholds = stage_thresholds[s_idx]

        if guide_idx in merged_labels:
            guide_labels = merged_labels[guide_idx].copy()
        else:
            guide_labels = levels[guide_idx]["labels"].copy()

        fine_labels = levels[fine_idx]["labels"].copy()

        guide_K = len(set(guide_labels[guide_labels >= 0].tolist()))
        fine_K = len(set(fine_labels[fine_labels >= 0].tolist()))

        log.info(f"  ── Stage {s_idx+1}: guide=L{guide_idx} (K={guide_K}) → fine=L{fine_idx} (K={fine_K})")

        stage_result = _cascade_merge_stage(
            X=X,
            guide_labels=guide_labels,
            fine_labels=fine_labels,
            adj_csr=adj_csr,
            k=k,
            cos_threshold=thresholds["cos"],
            edge_threshold=thresholds["edge"],
            strong_threshold=thresholds["strong"],
            homology_rescue=homology_rescue,
            homology_rescue_cos=homology_rescue_cos,
            homology_rescue_max_size=homology_rescue_max_size,
            logger=log,
        )

        merged_labels[fine_idx] = stage_result["merged_labels"]
        total_merges += stage_result["n_merges"]
        total_rejected += stage_result["n_rejected"]
        total_homology_rescues += stage_result.get("n_homology_rescues", 0)
        all_merge_log.extend(stage_result["merge_log"])

        result_K = len(set(stage_result["merged_labels"][stage_result["merged_labels"] >= 0].tolist()))
        log.info(
            f"    → Merges: {stage_result['n_merges']} "
            f"(rejected: {stage_result['n_rejected']}, "
            f"homology_rescue: {stage_result.get('n_homology_rescues', 0)}) → K={result_K}"
        )
        stage_summaries.append({
            "stage": s_idx + 1,
            "guide_level_idx": guide_idx,
            "fine_level_idx": fine_idx,
            "guide_k": guide_K,
            "fine_k_before": fine_K,
            "fine_k_after": result_K,
            "n_merges": stage_result["n_merges"],
            "n_rejected": stage_result["n_rejected"],
            "n_homology_rescues": stage_result.get("n_homology_rescues", 0),
            "cos_threshold": thresholds["cos"],
            "edge_threshold": thresholds["edge"],
            "strong_threshold": thresholds["strong"],
        })

    coarse_level_idx = selected[0]
    if coarse_level_idx in merged_labels:
        coarse_out, _ = relabel_contiguous(merged_labels[coarse_level_idx])
    else:
        coarse_out, _ = relabel_contiguous(levels[coarse_level_idx]["labels"].copy())

    fine_level_idx = selected[-1]
    fine_out, _ = relabel_contiguous(merged_labels[fine_level_idx])

    if n_stages >= 2:
        mid_idx = selected[len(selected) // 2]
        if mid_idx in merged_labels:
            mid_out, _ = relabel_contiguous(merged_labels[mid_idx])
        else:
            mid_out, _ = relabel_contiguous(levels[mid_idx]["labels"].copy())
    else:
        mid_out = fine_out.copy()

    # ── Cross-branch rescue pass ──
    n_cross_branch_merges = 0
    if cross_branch_rescue:
        cb_result = _cross_branch_rescue(
            X=X,
            fine_labels=fine_out,
            coarse_labels=coarse_out,
            adj_csr=adj_csr,
            k=k,
            cross_branch_cos=cross_branch_cos,
            cross_branch_edge=cross_branch_edge,
            max_merges=max_cross_branch_merges,
            logger=log,
        )
        fine_out = cb_result["labels"]
        fine_out, _ = relabel_contiguous(fine_out)
        n_cross_branch_merges = cb_result["n_merges"]
        total_merges += n_cross_branch_merges
        all_merge_log.extend(cb_result["merge_log"])
        log.info(f"  Cross-branch rescue: {n_cross_branch_merges} merges performed")

    g = snn_graph_result["graph"]
    adaptive_out = fine_out.copy()
    max_fine_id = adaptive_out.max() if adaptive_out.size > 0 else 0
    for coarse_cid in set(coarse_out[coarse_out >= 0].tolist()):
        coarse_mask = coarse_out == coarse_cid
        coarse_members = np.where(coarse_mask)[0]
        fine_ids_here = set(fine_out[coarse_mask][fine_out[coarse_mask] >= 0].tolist())

        if len(fine_ids_here) <= 1:
            continue

        if len(coarse_members) >= 2:
            sg_c = g.subgraph(coarse_members.tolist())
            possible_c = len(coarse_members) * (len(coarse_members) - 1) / 2
            density_c = sg_c.ecount() / possible_c if possible_c > 0 else 0.0
        else:
            density_c = 0.0

        fine_densities = []
        for fid in fine_ids_here:
            fm = np.where(fine_out == fid)[0]
            if len(fm) >= 2:
                sg_f = g.subgraph(fm.tolist())
                possible_f = len(fm) * (len(fm) - 1) / 2
                fine_densities.append(sg_f.ecount() / possible_f if possible_f > 0 else 0.0)
            else:
                fine_densities.append(0.0)
        density_f = float(np.mean(fine_densities)) if fine_densities else 0.0

        if density_c > density_f:
            for node in coarse_members:
                adaptive_out[node] = coarse_cid + max_fine_id + 1

    adaptive_out, _ = relabel_contiguous(adaptive_out)

    n_clusters_fine = len(set(fine_out[fine_out >= 0].tolist()))
    n_singletons = int(np.sum(fine_out == -1))

    log.info(f"  Total merge candidates evaluated: {total_merges + total_rejected}")
    log.info(f"  Total merges performed: {total_merges} (rejected: {total_rejected})")
    log.info(f"  Homology rescues: {total_homology_rescues}")
    log.info(f"  Cross-branch rescues: {n_cross_branch_merges}")
    log.info(f"  Final clusters (fine/merged): {n_clusters_fine} (+{n_singletons} singletons)")
    log.info(f"  Output level: {output_level}")

    if total_merges > 0 and total_rejected == 0:
        log.warning(f"  ⚠ ALL merge candidates were accepted (0 rejections).")
    total_input_clusters = sum(levels[selected[s+1]]["n_clusters"] for s in range(n_stages))
    if total_merges > total_input_clusters * 0.5:
        merge_pct = 100 * total_merges / max(total_input_clusters, 1)
        log.warning(f"  ⚠ HIGH MERGE RATE: {total_merges} merges ({merge_pct:.0f}% of input clusters).")

    labels_all = {
        "coarse": coarse_out,
        "fine": fine_out,
        "mid": mid_out,
        "adaptive": adaptive_out,
    }

    primary = labels_all.get(output_level, adaptive_out)

    return {
        "labels": primary,
        "labels_all": labels_all,
        "n_clusters": len(set(primary[primary >= 0].tolist())),
        "n_singletons": int(np.sum(primary == -1)),
        "merge_log": all_merge_log,
        "n_merges": total_merges,
        "stage_summaries": stage_summaries,
    }


# ---------------------------------------------------------------------------
#  Cross-branch rescue
# ---------------------------------------------------------------------------

def _cross_branch_rescue(
    X, fine_labels, coarse_labels, adj_csr, k,
    cross_branch_cos, cross_branch_edge, max_merges, logger=None,
):
    """
    After branch-local cascaded merge, attempt a single global pass that
    allows merges across coarse-level branch boundaries when both cosine
    similarity and edge connectivity evidence is strong.

    Only pairs whose coarse parents share a common grandparent (if available)
    or are direct coarse neighbors are considered.
    """
    log = logger or logging.getLogger("multiscale")
    log.info("  ── Cross-branch rescue pass ──")

    merged = fine_labels.copy()
    merge_log = []
    n_merges = 0

    # Build cluster info
    cluster_info = {}
    for fid in set(merged[merged >= 0].tolist()):
        members = np.where(merged == fid)[0]
        cluster_info[fid] = {
            "members": set(members.tolist()),
            "sum": X[members].sum(axis=0),
            "count": len(members),
        }

    # Build cross-edge counts and neighbor map
    node_to_cluster = {}
    for fid, info in cluster_info.items():
        for node in info["members"]:
            node_to_cluster[node] = fid

    cross_edges = defaultdict(int)
    cluster_neighbors = defaultdict(set)

    for node_i in range(adj_csr.shape[0]):
        ci = node_to_cluster.get(node_i, -1)
        if ci < 0:
            continue
        row_start = adj_csr.indptr[node_i]
        row_end = adj_csr.indptr[node_i + 1]
        for idx in range(row_start, row_end):
            node_j = adj_csr.indices[idx]
            cj = node_to_cluster.get(node_j, -1)
            if cj < 0 or ci == cj:
                continue
            key = (min(ci, cj), max(ci, cj))
            cross_edges[key] += 1
            cluster_neighbors[ci].add(cj)
            cluster_neighbors[cj].add(ci)

    for key in cross_edges:
        cross_edges[key] //= 2

    # Build coarse branch mapping: fine_cluster -> coarse_cluster
    fine_to_coarse = {}
    for fid in cluster_info:
        members = list(cluster_info[fid]["members"])
        if members:
            coarse_votes = coarse_labels[members]
            coarse_votes = coarse_votes[coarse_votes >= 0]
            if len(coarse_votes) > 0:
                fine_to_coarse[fid] = int(np.argmax(np.bincount(coarse_votes)))

    # Score all cross-branch pairs
    candidates = []
    seen_pairs = set()

    for fi in list(cluster_info.keys()):
        coarse_fi = fine_to_coarse.get(fi, -1)
        for fj in cluster_neighbors.get(fi, set()):
            if fj <= fi:
                continue
            pair = (fi, fj)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            coarse_fj = fine_to_coarse.get(fj, -1)
            # Only consider CROSS-branch pairs
            if coarse_fi == coarse_fj and coarse_fi >= 0:
                continue

            # Cosine similarity
            ci_vec = cluster_info[fi]["sum"] / max(cluster_info[fi]["count"], 1)
            cj_vec = cluster_info[fj]["sum"] / max(cluster_info[fj]["count"], 1)
            ni, nj = np.linalg.norm(ci_vec), np.linalg.norm(cj_vec)
            cos_sim = float(np.dot(ci_vec, cj_vec) / (ni * nj + EPS))

            # Edge connectivity
            key = (min(fi, fj), max(fi, fj))
            n_cross = cross_edges.get(key, 0)
            min_size = min(cluster_info[fi]["count"], cluster_info[fj]["count"])
            denom = min_size * k if k > 0 else min_size
            edge_conn = n_cross / denom if denom > 0 else 0.0

            if cos_sim >= cross_branch_cos and edge_conn >= cross_branch_edge:
                score = 0.7 * edge_conn + 0.3 * cos_sim
                candidates.append((score, fi, fj, cos_sim, edge_conn))

    # Sort by score descending, merge top candidates
    candidates.sort(key=lambda x: -x[0])
    cluster_version = {fid: 0 for fid in cluster_info}

    for score, fi, fj, cos_sim, edge_conn in candidates:
        if n_merges >= max_merges:
            break
        if fi not in cluster_info or fj not in cluster_info:
            continue

        # Merge fj into fi
        info_i = cluster_info[fi]
        info_j = cluster_info.pop(fj)

        info_i["members"].update(info_j["members"])
        info_i["sum"] = info_i["sum"] + info_j["sum"]
        info_i["count"] += info_j["count"]

        for node in info_j["members"]:
            node_to_cluster[node] = fi
            merged[node] = fi

        merge_log.append({
            "child_a": int(fi), "child_b": int(fj),
            "centroid_sim": cos_sim, "edge_connectivity": edge_conn,
            "merged": True,
            "reason": f"cross_branch cos={cos_sim:.3f}>={cross_branch_cos}, edge={edge_conn:.3f}>={cross_branch_edge}",
            "merge_source": "cross_branch",
        })
        n_merges += 1

    log.info(f"    Cross-branch candidates: {len(candidates)}, merges: {n_merges}")

    return {
        "labels": merged,
        "n_merges": n_merges,
        "merge_log": merge_log,
    }


# ---------------------------------------------------------------------------
#  Branch-local cascade merge stage (with homology rescue)
# ---------------------------------------------------------------------------

def _cascade_merge_stage(
    X, guide_labels, fine_labels, adj_csr, k,
    cos_threshold, edge_threshold, strong_threshold,
    homology_rescue=False, homology_rescue_cos=0.95, homology_rescue_max_size=20,
    logger=None,
):
    log = logger or logging.getLogger("multiscale")
    N = len(fine_labels)
    merged = fine_labels.copy()
    merge_log = []
    n_merges = 0
    n_rejected = 0
    n_homology_rescues = 0

    cluster_info = {}
    for fid in set(merged[merged >= 0].tolist()):
        members = np.where(merged == fid)[0]
        cluster_info[fid] = {
            "members": set(members.tolist()),
            "sum": X[members].sum(axis=0),
            "count": len(members),
        }

    cross_edges = defaultdict(int)
    cluster_neighbors = defaultdict(set)

    node_to_cluster = {}
    for fid, info in cluster_info.items():
        for node in info["members"]:
            node_to_cluster[node] = fid

    for node_i in range(adj_csr.shape[0]):
        ci = node_to_cluster.get(node_i, -1)
        if ci < 0:
            continue
        row_start = adj_csr.indptr[node_i]
        row_end = adj_csr.indptr[node_i + 1]
        for idx in range(row_start, row_end):
            node_j = adj_csr.indices[idx]
            cj = node_to_cluster.get(node_j, -1)
            if cj < 0 or ci == cj:
                continue
            key = (min(ci, cj), max(ci, cj))
            cross_edges[key] += 1
            cluster_neighbors[ci].add(cj)
            cluster_neighbors[cj].add(ci)

    for key in cross_edges:
        cross_edges[key] //= 2

    cluster_version = {fid: 0 for fid in cluster_info}
    all_sizes = [info["count"] for info in cluster_info.values()]
    median_size = float(np.median(all_sizes)) if all_sizes else 1.0
    size_threshold = max(2.0 * median_size, 10)

    def _edge_conn(fi, fj):
        key = (min(fi, fj), max(fi, fj))
        n_cross = cross_edges.get(key, 0)
        min_size = min(cluster_info[fi]["count"], cluster_info[fj]["count"])
        denom = min_size * k if k > 0 else min_size
        return n_cross, n_cross / denom if denom > 0 else 0.0

    def _centroid_cos(fi, fj):
        ci_vec = cluster_info[fi]["sum"] / max(cluster_info[fi]["count"], 1)
        cj_vec = cluster_info[fj]["sum"] / max(cluster_info[fj]["count"], 1)
        ni, nj = np.linalg.norm(ci_vec), np.linalg.norm(cj_vec)
        return float(np.dot(ci_vec, cj_vec) / (ni * nj + EPS))

    def _score_pair(fi, fj):
        _, edge_conn = _edge_conn(fi, fj)
        cos_sim = _centroid_cos(fi, fj)

        combined_size = cluster_info[fi]["count"] + cluster_info[fj]["count"]
        if combined_size > size_threshold:
            eff_cos = cos_threshold + 0.05
            eff_edge = edge_threshold * 1.5
            eff_strong = strong_threshold * 1.5
        else:
            eff_cos = cos_threshold
            eff_edge = edge_threshold
            eff_strong = strong_threshold

        passes = False
        reason = ""
        merge_source = "branch_local"

        # Path A: strong edge connectivity alone
        if edge_conn >= eff_strong:
            passes = True
            reason = f"strong edge_conn {edge_conn:.3f} >= {eff_strong:.3f}"
        # Path B: cosine + edge
        elif cos_sim >= eff_cos and edge_conn >= eff_edge:
            passes = True
            reason = f"cos {cos_sim:.3f}>={eff_cos:.3f} & edge {edge_conn:.3f}>={eff_edge:.3f}"
        # Path C: homology rescue — very high cosine for small clusters
        elif (
            homology_rescue
            and cos_sim >= homology_rescue_cos
            and cluster_info[fi]["count"] <= homology_rescue_max_size
            and cluster_info[fj]["count"] <= homology_rescue_max_size
        ):
            passes = True
            reason = (
                f"homology_rescue cos={cos_sim:.3f}>={homology_rescue_cos:.3f} "
                f"(sizes {cluster_info[fi]['count']},{cluster_info[fj]['count']})"
            )
            merge_source = "homology_rescue"

        score = 0.7 * edge_conn + 0.3 * cos_sim if passes else 0.0
        return passes, score, cos_sim, edge_conn, reason, merge_source

    def _merge_clusters(fi, fj):
        info_i = cluster_info[fi]
        info_j = cluster_info.pop(fj)

        info_i["members"].update(info_j["members"])
        info_i["sum"] = info_i["sum"] + info_j["sum"]
        info_i["count"] += info_j["count"]

        for node in info_j["members"]:
            node_to_cluster[node] = fi
            merged[node] = fi

        neighbors_j = cluster_neighbors.pop(fj, set())
        for other in neighbors_j:
            if other == fi:
                continue
            old_key = (min(fj, other), max(fj, other))
            count = cross_edges.get(old_key, 0)
            if count > 0:
                new_key = (min(fi, other), max(fi, other))
                cross_edges[new_key] += count
                cluster_neighbors[fi].add(other)
                cluster_neighbors[other].discard(fj)
                cluster_neighbors[other].add(fi)

        cross_edges.pop((min(fi, fj), max(fi, fj)), None)
        cluster_neighbors[fi].discard(fj)

        cluster_version[fi] = cluster_version.get(fi, 0) + 1
        cluster_version.pop(fj, None)

    guide_ids = sorted(set(guide_labels[guide_labels >= 0].tolist()))

    for guide_cid in guide_ids:
        guide_mask = guide_labels == guide_cid
        fine_ids_in_branch = sorted(set(merged[guide_mask & (merged >= 0)].tolist()))

        if len(fine_ids_in_branch) <= 1:
            continue

        branch_set = set(fine_ids_in_branch)
        heap = []

        scored_pairs = set()
        for fi in fine_ids_in_branch:
            # Score against graph neighbors within this branch
            for fj in cluster_neighbors.get(fi, set()):
                if fj not in branch_set or fj <= fi:
                    continue
                pair_key = (fi, fj)
                if pair_key in scored_pairs:
                    continue
                scored_pairs.add(pair_key)

                passes, score, cos_sim, edge_conn, reason, merge_source = _score_pair(fi, fj)
                if not passes:
                    merge_log.append({
                        "child_a": int(fi), "child_b": int(fj),
                        "centroid_sim": cos_sim, "edge_connectivity": edge_conn,
                        "merged": False, "reason": reason or "below thresholds",
                        "merge_source": merge_source,
                    })
                    n_rejected += 1
                    continue

                vi = cluster_version.get(fi, 0)
                vj = cluster_version.get(fj, 0)
                heapq.heappush(heap, (-score, fi, fj, vi, vj, cos_sim, edge_conn, reason, merge_source))

            # Homology rescue: also check non-neighbor siblings in the branch
            if homology_rescue:
                for fj in fine_ids_in_branch:
                    if fj <= fi or fj not in cluster_info or fi not in cluster_info:
                        continue
                    pair_key = (fi, fj)
                    if pair_key in scored_pairs:
                        continue
                    # Only attempt for small clusters
                    if (cluster_info[fi]["count"] > homology_rescue_max_size
                            or cluster_info[fj]["count"] > homology_rescue_max_size):
                        continue

                    cos_sim = _centroid_cos(fi, fj)
                    if cos_sim >= homology_rescue_cos:
                        scored_pairs.add(pair_key)
                        _, edge_conn = _edge_conn(fi, fj)
                        reason = (
                            f"homology_rescue cos={cos_sim:.3f}>={homology_rescue_cos:.3f} "
                            f"(sizes {cluster_info[fi]['count']},{cluster_info[fj]['count']})"
                        )
                        score = 0.7 * edge_conn + 0.3 * cos_sim
                        vi = cluster_version.get(fi, 0)
                        vj = cluster_version.get(fj, 0)
                        heapq.heappush(heap, (-score, fi, fj, vi, vj, cos_sim, edge_conn, reason, "homology_rescue"))

        while heap:
            neg_score, fi, fj, vi, vj, cos_sim, edge_conn, reason, merge_source = heapq.heappop(heap)

            if fi not in cluster_info or fj not in cluster_info:
                continue
            if cluster_version.get(fi, -1) != vi or cluster_version.get(fj, -1) != vj:
                continue

            _merge_clusters(fi, fj)
            branch_set.discard(fj)
            merge_log.append({
                "child_a": int(fi), "child_b": int(fj),
                "centroid_sim": cos_sim, "edge_connectivity": edge_conn,
                "merged": True, "reason": reason,
                "merge_source": merge_source,
            })
            n_merges += 1
            if merge_source == "homology_rescue":
                n_homology_rescues += 1

            for other in list(cluster_neighbors.get(fi, set())):
                if other not in branch_set or other not in cluster_info:
                    continue
                passes2, score2, cos2, edge2, reason2, source2 = _score_pair(fi, other)
                if passes2:
                    vi2 = cluster_version.get(fi, 0)
                    vo2 = cluster_version.get(other, 0)
                    heapq.heappush(heap, (-score2, fi, other, vi2, vo2, cos2, edge2, reason2, source2))

    return {
        "merged_labels": merged,
        "n_merges": n_merges,
        "n_rejected": n_rejected,
        "n_homology_rescues": n_homology_rescues,
        "merge_log": merge_log,
    }
