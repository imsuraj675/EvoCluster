import logging
import numpy as np
from collections import Counter
from .evaluation import relabel_contiguous

def run_pygenstability_discovery(snn_graph_result, logger=None):
    """Run PyGenStability to find robust topological partitions as candidate scales."""
    log = logger or logging.getLogger("multiscale")
    try:
        from pygenstability import run
    except ImportError:
        log.error("PyGenStability is not installed. Please install it with `pip install pygenstability`.")
        raise
    
    log.info("Step 3 (PyGenStability): Discovering candidate stable regions via Markov Time sweeps...")
    
    # Extract the Scipy Sparse Adjacency Matrix
    adj = snn_graph_result["adjacency"].astype(np.float64)
    N = snn_graph_result["n_nodes"]
    
    from scipy.sparse.csgraph import connected_components
    n_comp, comp_labels = connected_components(adj, directed=False)
    
    if n_comp > 1:
        log.warning(f"  ⚠ Graph has {n_comp} disconnected components. PyGenStability requires a single component.")
        log.info("  Extracting the largest connected component for Markov stability sweep...")
        largest_comp_id = np.argmax(np.bincount(comp_labels))
        mask = comp_labels == largest_comp_id
        adj_sub = adj[mask][:, mask].copy()
        
        # 2.1 Adjacency Not Markov-Compatible Fix: Row normalization
        row_sums = np.array(adj_sub.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        adj_sub = adj_sub.multiply(1.0 / row_sums[:, None])
        
        # Run the continuous-time Markov stability sweep
        results = run(adj_sub)
    else:
        mask = np.ones(N, dtype=bool)
        
        # 2.1 Adjacency Not Markov-Compatible Fix: Row normalization
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        adj = adj.multiply(1.0 / row_sums[:, None])
        
        # Run the continuous-time Markov stability sweep
        results = run(adj)
    
    levels = []
    
    # 3. API Robustness Fix for PyGenStability v0.2.3+
    # Results is a monolithic dict containing arrays for all tested scales
    if isinstance(results, dict) and "community_id" in results and "scales" in results:
        scales = results["scales"]
        community_id = results["community_id"]
        
        # TASK A: Extract all distinct candidate partitions, ignoring the "robust only" flag
        indices = np.arange(len(scales))
        log.info(f"  PyGen stability yielded {len(community_id)} total partitions. Extracting all distinct candidates...")
        
        last_n_clusters = -1
        last_labels = None
        
        for i in indices:
            scale_val = scales[i]
            sub_labels = np.array(community_id[i], dtype=np.int32)
            
            # Map subset labels back to the full graph
            full_labels = np.full(N, -1, dtype=np.int32)
            full_labels[mask] = sub_labels
            
            # 2.5 Label Normalization
            full_labels, _ = relabel_contiguous(full_labels)
            n_clusters = len(np.unique(full_labels[full_labels >= 0]))
            
            # Deduplication: block if mathematically identical to the previous scale
            if n_clusters == last_n_clusters and np.array_equal(full_labels, last_labels):
                continue
            
            last_n_clusters = n_clusters
            last_labels = full_labels.copy()
            
            # 2.4 Feasibility Filtering
            if n_clusters > 1 and n_clusters < 0.8 * N:
                levels.append({
                    "resolution": float(scale_val),
                    "n_clusters": n_clusters,
                    "labels": full_labels,
                    "sizes": Counter(full_labels[full_labels >= 0].tolist())
                })
                log.info(f"  ▸ PyGen candidate scale {float(scale_val):.4f} → K={n_clusters}")
                
        # TASK A: Explicitly sort levels from coarse to fine (K ascending)
        levels = sorted(levels, key=lambda x: x["n_clusters"])
        for i, lvl in enumerate(levels):
            log.info(f"  Level {i}: K={lvl['n_clusters']} (Resolution: {lvl['resolution']:.4f})")
    else:
        log.error("Unknown PyGenStability result format (expected dict with 'scales' and 'community_id')")
        
    if not levels:
        log.warning("  ⚠ PyGen stability returned no feasible scales.")

    # Only return the equivalent of ms_result (levels array), no fake scoring
    return {
        "levels": levels,
    }

def run_resolution_profile_discovery(snn_graph_result, resolution_range=(0.001, 1.0), objective_function="CPM", logger=None):
    """Run leidenalg Resolution Profile to find structural graph transitions."""
    log = logger or logging.getLogger("multiscale")
    try:
        import leidenalg as la
    except ImportError:
        log.error("leidenalg is not installed. Please install it with `pip install leidenalg`.")
        raise
        
    g = snn_graph_result["graph"]
    weights = g.es['weight'] if 'weight' in g.edge_attributes() else None
    
    if objective_function.upper() == "CPM":
        partition_type = la.CPMVertexPartition
    else:
        partition_type = la.RBConfigurationVertexPartition
        
    log.info(f"Step 3 (Resolution Profile): Scanning discrete graph transitions across resolution span {resolution_range}...")
    
    optimiser = la.Optimiser()
    profile = optimiser.resolution_profile(g, partition_type, 
                                           resolution_range=resolution_range, 
                                           weights=weights)
                                           
    # 4.2 Missing Sorting Fix
    profile = sorted(profile, key=lambda p: getattr(p, 'resolution_parameter', 0.0))
    
    levels = []
    
    N = snn_graph_result["n_nodes"]
    
    # 4.3 Missing Logging Fix
    log.info("  Resolution profile curve before filtering:")
    for idx, part in enumerate(profile):
        labels = np.array(part.membership, dtype=np.int32)
        labels, _ = relabel_contiguous(labels)
        n_clusters = len(np.unique(labels[labels >= 0]))
        res = getattr(part, 'resolution_parameter', float(idx))
        
        log.info(f"    res={res:.4f} → K={n_clusters}")
        
        # Immediate Feasibility Filtering
        if n_clusters > 1 and n_clusters < 0.8 * N:
            levels.append({
                "resolution": res,
                "n_clusters": n_clusters,
                "labels": labels,
                "sizes": Counter(labels[labels >= 0].tolist())
            })
            
    # TASK B: Thinning Resolution Profile
    # The profile can generate hundreds of near-identical partitions.
    # We thin them by requiring at least a 5% step in K between retained candidates.
    selected_levels = []
    if levels:
        # Always keep the coarsest feasible level
        selected_levels.append(levels[0])
        last_k = levels[0]["n_clusters"]
        
        for lvl in levels[1:]:
            step_threshold = max(2, int(last_k * 0.05))
            if lvl["n_clusters"] >= last_k + step_threshold:
                selected_levels.append(lvl)
                last_k = lvl["n_clusters"]
                
        # Always keep the finest feasible level if distinct
        if levels[-1]["resolution"] != selected_levels[-1]["resolution"]:
            # Only add if it hasn't already been grabbed
            if levels[-1]["n_clusters"] > selected_levels[-1]["n_clusters"]:
                selected_levels.append(levels[-1])
                
    # TASK A: Explicitly sort levels from coarse to fine (K ascending)
    selected_levels = sorted(selected_levels, key=lambda x: x["n_clusters"])
                
    log.info(f"  ▸ Profile thinned to {len(selected_levels)} candidate scales based on K-spacing.")
    for lvl in selected_levels:
        log.info(f"    Selected candidate: res={lvl['resolution']:.4f} → K={lvl['n_clusters']}")

    return {
        "levels": selected_levels,
    }
