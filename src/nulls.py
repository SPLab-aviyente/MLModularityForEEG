import random
import copy

from .mlgraph import MLGraph

def weight_preserved(G: MLGraph, n_swaps: int=5, preserve_layer: bool=False) -> MLGraph:
    """Generate a null weighted multilayer graph from a given weighted multilayer
    graph by swapping its edges. The method selects an edge (a, b, w_ab) and 
    two nodes c and d; where w_ab is the edge weight and a, b, c, and d are all 
    distinct. If (c, d, w_cd) is not an edge, it removes (a, b, w_ab) and add 
    the edge (c, d, w_ab) to the graph. If (c, d, w_cd) is an edge, it swaps
    edge weights of two edges. Check [1] for more details. If preserve layer is 
    True, it is ensured that layers of a and c and layers of b and d are the same.

    Parameters
    ----------
    G : MLGraph
        The multilayer graph from which the null multilayer graph will be generated.
    n_swaps : int, optional
        Number of times to swap each edge, by default 5
    preserve_layer : bool, optional
        Whether to preserve layer of the edge when swapping it, by default False

    Returns
    -------
    Gn: MLGraph
        Null multilayer graph.

    Raises
    ------
    Exception
        Input graph must be weighted.

    References
    ----------
    .. [1] Ansmann, Gerrit, and Klaus Lehnertz. "Constrained randomization of 
           weighted networks." Physical Review E 84.2 (2011): 026103.	
    """

    n_nodes = G.order()
    n_edges = G.size()

    if not G.graph.is_weighted():
        raise Exception("Provided graph must be weighted.")

    Gc = copy.deepcopy(G)

    node_idx = Gc.graph.vs.indices

    if preserve_layer:
        layers = Gc.layers
        node_idx_layer = {}
        for layer in layers:
            node_idx_layer[layer] = G.layer_vertices(layer).indices

    n_swaps *= n_edges
    max_attempt = int(n_nodes*n_edges/(n_nodes*(n_nodes-1)))

    for _ in range(n_swaps):
        attempt = 0
        while attempt <= max_attempt:
            attempt += 1

            # select and edge to swap
            a = random.choice(node_idx)

            if len(Gc.graph.neighbors(a)) == 0:
                continue

            b = random.choice(Gc.graph.neighbors(a))

            # select a pair of nodes different than (a, b) to consider for swap with (a, b)
            if preserve_layer:
                la = Gc.graph.vs[a]["layer"]
                lb = Gc.graph.vs[b]["layer"]
                c = random.choice(node_idx_layer[la])
                d = random.choice(node_idx_layer[lb])
            else:
                c = random.choice(node_idx)
                d = random.choice(node_idx)

            if c == d:
                continue

            if (c==b or c==a) and (d==b or d==a):
                continue

            # rewire
            edge_ab_id = Gc.graph.get_eid(a, b)
            edge_cd_id = Gc.graph.get_eid(c, d, error=False) # returns -1, if (c, d) is not an edge

            w_ab = Gc.graph.es[edge_ab_id]["weight"]
            if edge_cd_id == -1:
                Gc.graph.delete_edges(edge_ab_id)
                Gc.graph.add_edge(c, d, weight=w_ab)
                break
            else:
                w_cd = Gc.graph.es[edge_cd_id]["weight"]

                if w_ab != w_cd:
                    Gc.graph.es[edge_ab_id]["weight"] = w_cd
                    Gc.graph.es[edge_cd_id]["weight"] = w_ab
                    break

    return Gc