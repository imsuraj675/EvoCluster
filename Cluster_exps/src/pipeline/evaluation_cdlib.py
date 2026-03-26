import logging
import numpy as np

def evaluate_topology_cdlib(graph, labels, logger=None):
    """
    Evaluates topological partition quality using CDlib.
    Requires `graph` (igraph.Graph) and `labels` (array of length N).
    """
    log = logger or logging.getLogger("multiscale")
    metrics = {}
    
    try:
        from cdlib import NodeClustering, evaluation
    except ImportError:
        log.debug("cdlib is not installed. Skipping advanced topological metrics.")
        return metrics
        
    if graph is None:
        return metrics

    # Map contiguous labels array into list of node indices per community
    community_dict = {}
    for node_idx, cluster_id in enumerate(labels):
        if cluster_id >= 0:  # Ignore -1 singletons for community topological benchmarking
            if cluster_id not in community_dict:
                community_dict[cluster_id] = []
            community_dict[cluster_id].append(node_idx)
            
    communities = list(community_dict.values())
    
    if not communities:
        return metrics
        
    try:
        # 0. Hotfix for CDlib: ensure standard attributes exist on igraph
        if "name" not in graph.vs.attributes():
            graph.vs["name"] = [i for i in range(graph.vcount())]
            
        if "weight" not in graph.es.attributes():
            graph.es["weight"] = [1.0] * graph.ecount()
            
        # Create CDlib NodeClustering object linking the igraph and community sequence
        clustering = NodeClustering(communities, graph=graph, method_name="EvoCluster")
        
        # 1. Modularity (Newman-Girvan)
        # Difference between edges within communities and random expectations
        mod = evaluation.newman_girvan_modularity(graph, clustering)
        metrics["modularity"] = mod.score
        
        # 2. Conductance (Average)
        # Ratio of edges outside the cluster to the edges inside (lower is better structural isolation)
        cond = evaluation.conductance(graph, clustering)
        metrics["conductance"] = cond.score
        
        # 3. Scaled Density
        # Internal density scaled by size
        dens = evaluation.scaled_density(graph, clustering)
        metrics["density"] = dens.score
        
        # 4. Intra-Cluster Density
        intra = evaluation.internal_edge_density(graph, clustering)
        metrics["intra_density"] = intra.score
        
        # 5. Coverage
        # Fraction of nodes covered by the communities (this is basically 1.0 minus singleton fraction)
        cov = evaluation.coverage(graph, clustering)
        metrics["coverage"] = cov.score
        
    except Exception as e:
        import traceback
        log.debug(f"CDlib topological evaluation failed: {e}")
        log.debug("Traceback Details:\n" + traceback.format_exc())
        
    return metrics
