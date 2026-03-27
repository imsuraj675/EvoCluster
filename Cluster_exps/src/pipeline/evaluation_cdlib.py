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

        # Coverage is not exposed as evaluation.coverage in the installed CDlib
        # version; NodeClustering already computes node_coverage internally.
        metrics["coverage"] = float(getattr(clustering, "node_coverage", 0.0))

        metric_fns = {
            "modularity": evaluation.newman_girvan_modularity,
            "conductance": evaluation.conductance,
            "density": evaluation.scaled_density,
            "intra_density": evaluation.internal_edge_density,
        }

        for metric_name, metric_fn in metric_fns.items():
            try:
                result = metric_fn(graph, clustering)
                metrics[metric_name] = float(result.score)
            except Exception as metric_error:
                log.debug(f"CDlib metric '{metric_name}' failed: {metric_error}")

    except Exception as e:
        import traceback
        log.debug(f"CDlib topological evaluation failed: {e}")
        log.debug("Traceback Details:\n" + traceback.format_exc())
        
    return metrics
