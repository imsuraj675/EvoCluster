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
    k_neighbors=None,
    homology_rescue=False,
    homology_rescue_cos=0.95,
    homology_rescue_max_size=20,
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

    adj_csr = adjacency.tocsr() if not hasattr(adjacency, 'indptr') else adjacency

    n_stages = len(selected) - 1
    
    if len(selected) == 0:
        log.error("  CRITICAL: 0 levels selected for merge cascade! Returning all singletons.")
        single_labels = np.full(N, -1, dtype=np.int32)
        return {
            "labels": single_labels,
            "labels_all": {"fine": single_labels},
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
            "labels_all": {"fine": single_out},
            "n_clusters": len(set(single_out[single_out >= 0].tolist())),
            "n_singletons": int(np.sum(single_out == -1)),
            "merge_log": [],
            "n_merges": 0,
            "stage_summaries": [],
        }

    # Escalation magnitudes: controls how much thresholds relax from the first
    # cascade stage (strictest) to the last (loosest).  Quadratic ramp means
    # early stages stay very close to the base threshold while the final stage
    # gets the full relaxation budget.  Values were empirically tuned on
    # pfal_pber / mmus_hsap to keep precision drops < 5% at max escalation.
    COS_ESCALATION  = 0.10   # allows up to +10% cosine relaxation at final stage
    EDGE_ESCALATION = 0.05   # allows up to +5% edge-connectivity relaxation
    STRONG_ESCALATION = 0.05 # same for the strong-edge shortcut path

    stage_thresholds = []
    for s_idx in range(n_stages):
        t_linear = s_idx / max(n_stages - 1, 1) if n_stages > 1 else 0.0
        # Quadratic escalation for modest early-stage loosening
        t_quad = t_linear ** 2
        
        cos_t  = centroid_cos_threshold    + COS_ESCALATION * t_quad
        edge_t = edge_connectivity_threshold + EDGE_ESCALATION * t_quad
        strong_t = edge_strong_threshold + STRONG_ESCALATION * t_quad
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

        is_final_stage = (s_idx == n_stages - 1)
        X_centroids = X

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
            X_centroids=X_centroids,
            is_final_stage=is_final_stage,
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
            f"homology_rescue: {stage_result.get('n_homology_rescues', 0)}, "
            f"size_rescue: {stage_result.get('n_size_rescues', 0)}) → K={result_K}"
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
            "n_size_rescues": stage_result.get("n_size_rescues", 0),
            "cos_threshold": thresholds["cos"],
            "edge_threshold": thresholds["edge"],
            "strong_threshold": thresholds["strong"],
        })

    fine_level_idx = selected[-1]
    fine_out, _ = relabel_contiguous(merged_labels[fine_level_idx])

    n_clusters_fine = len(set(fine_out[fine_out >= 0].tolist()))
    n_singletons = int(np.sum(fine_out == -1))

    log.info(f"  Total merge candidates evaluated: {total_merges + total_rejected}")
    log.info(f"  Total merges performed: {total_merges} (rejected: {total_rejected})")
    log.info(f"  Homology rescues: {total_homology_rescues}")
    log.info(f"  Final clusters (fine/merged): {n_clusters_fine} (+{n_singletons} singletons)")

    if total_merges > 0 and total_rejected == 0:
        log.warning(f"  ⚠ ALL merge candidates were accepted (0 rejections).")
    total_input_clusters = sum(levels[selected[s+1]]["n_clusters"] for s in range(n_stages))
    if total_merges > total_input_clusters * 0.5:
        merge_pct = 100 * total_merges / max(total_input_clusters, 1)
        log.warning(f"  ⚠ HIGH MERGE RATE: {total_merges} merges ({merge_pct:.0f}% of input clusters).")

    primary = fine_out
    labels_all = {"fine": primary}

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
#  Branch-local cascade merge stage (with homology rescue)
# ---------------------------------------------------------------------------

def _cascade_merge_stage(
    X, guide_labels, fine_labels, adj_csr, k,
    cos_threshold, edge_threshold, strong_threshold,
    homology_rescue=False, homology_rescue_cos=0.95, homology_rescue_max_size=20,
    X_centroids=None, is_final_stage=False,
    logger=None,
):
    log = logger or logging.getLogger("multiscale")
    N = len(fine_labels)
    merged = fine_labels.copy()
    merge_log = []
    n_merges = 0
    n_rejected = 0
    n_homology_rescues = 0
    n_size_rescues = 0

    # X_centroids defaults to X if not provided (non-diffused)
    if X_centroids is None:
        X_centroids = X

    cluster_info = {}
    for fid in set(merged[merged >= 0].tolist()):
        members = np.where(merged == fid)[0]
        cluster_info[fid] = {
            "members": set(members.tolist()),
            "sum": X[members].sum(axis=0),
            "sum_centroids": X_centroids[members].sum(axis=0),
            "count": len(members),
        }

    cross_edges = defaultdict(int)
    cluster_neighbors = defaultdict(set)

    # Build node→cluster lookup (dict for merge ops, array for vectorized scan)
    node_to_cluster_arr = np.full(adj_csr.shape[0], -1, dtype=np.int64)
    node_to_cluster = {}
    for fid, info in cluster_info.items():
        for node in info["members"]:
            node_to_cluster[node] = fid
            node_to_cluster_arr[node] = fid

    # Vectorized cross-edge computation via COO representation
    coo = adj_csr.tocoo()
    ci_arr = node_to_cluster_arr[coo.row]
    cj_arr = node_to_cluster_arr[coo.col]
    # Filter to cross-cluster edges (both assigned, different clusters)
    cross_mask = (ci_arr >= 0) & (cj_arr >= 0) & (ci_arr != cj_arr)
    ci_cross = ci_arr[cross_mask]
    cj_cross = cj_arr[cross_mask]

    for ci_val, cj_val in zip(ci_cross.tolist(), cj_cross.tolist()):
        key = (min(ci_val, cj_val), max(ci_val, cj_val))
        cross_edges[key] += 1
        cluster_neighbors[ci_val].add(cj_val)
        cluster_neighbors[cj_val].add(ci_val)

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
        """Centroid cosine using X_centroids (possibly diffused)."""
        ci_vec = cluster_info[fi]["sum_centroids"] / max(cluster_info[fi]["count"], 1)
        cj_vec = cluster_info[fj]["sum_centroids"] / max(cluster_info[fj]["count"], 1)
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
        # Path D: size-aware final rescue — ratio-based check, final stage only
        elif (
            is_final_stage
            and homology_rescue
            and cos_sim >= homology_rescue_cos
        ):
            size_a = cluster_info[fi]["count"]
            size_b = cluster_info[fj]["count"]
            size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1)
            if size_ratio <= 3.0 or max(size_a, size_b) <= 10:
                passes = True
                reason = (
                    f"size_rescue cos={cos_sim:.3f}>={homology_rescue_cos:.3f} "
                    f"ratio={size_ratio:.1f} (sizes {size_a},{size_b})"
                )
                merge_source = "size_rescue"

        if passes:
            if merge_source in ("homology_rescue", "size_rescue"):
                # Prioritize rescue paths to execute before local clusters swell in size
                score = 1.0 + cos_sim 
            else:
                score = 0.7 * edge_conn + 0.3 * cos_sim
        else:
            score = 0.0
        return passes, score, cos_sim, edge_conn, reason, merge_source

    def _merge_clusters(fi, fj):
        info_i = cluster_info[fi]
        info_j = cluster_info.pop(fj)

        info_i["members"].update(info_j["members"])
        info_i["sum"] = info_i["sum"] + info_j["sum"]
        info_i["sum_centroids"] = info_i["sum_centroids"] + info_j["sum_centroids"]
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
            elif merge_source == "size_rescue":
                n_size_rescues += 1

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
        "n_size_rescues": n_size_rescues,
        "merge_log": merge_log,
    }
