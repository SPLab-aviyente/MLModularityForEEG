import random

from .. import mlgraph

def weight_preserved(G, n_swaps=5, preserve_layer=False):
    # TODO: Docstring

    n_nodes = G.vcount()
    n_edges = G.ecount()

    if not G.is_weighted():
        raise Exception("Provided graph must be weighted.")

    Gc = G.copy()

    # check if the provided graph is multilayer
    is_multilayer = True
    if not ("layer" in Gc.vertex_attributes()): # G is not multilayer
        is_multilayer = False
        preserve_layer = False

    node_idx = [v.index for v in Gc.vs]

    if preserve_layer:
        layers = mlgraph.get_layer_names(Gc)
        node_idx_layer = {}
        for layer in layers:
            node_idx_layer[layer] = [v.index for v in Gc.vs.select(layer_eq = layer)]

    n_swaps *= n_edges

    max_attempt = int(n_nodes*n_edges/(n_nodes*(n_nodes-1)))

    for _ in range(n_swaps):
        attempt = 0
        while attempt <= max_attempt:

            # select and edge to swap
            a = random.choice(node_idx)
            b = random.choice(Gc.neighbors(a))

            # select a pair of nodes different than (a, b) to consider for swap with (a, b)
            if preserve_layer:
                la = Gc.vs[a]["layer"]
                lb = Gc.vs[b]["layer"]
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
            edge_ab_id = Gc.get_eid(a, b)
            edge_cd_id = Gc.get_eid(c, d, error=False) # returns -1, if (c, d) is not an edge

            w_ab = Gc.es[edge_ab_id]["weight"]
            if edge_cd_id == -1:
                Gc.delete_edges(edge_ab_id)
                Gc.add_edge(c, d, weight=w_ab)
                break
            else:
                w_cd = Gc.es[edge_cd_id]["weight"]

                if w_ab != w_cd:
                    Gc.es[edge_ab_id]["weight"] = w_cd
                    Gc.es[edge_cd_id]["weight"] = w_ab
                    break

            attempt += 1

    return Gc

def strength_preserved(G, n_swaps=5, preserve_layer=False):
    # TODO: Docstring

    n_nodes = G.vcount()
    n_edges = G.ecount()

    if not G.is_weighted():
        raise Exception("Provided graph must be weighted.")

    # check if the provided graph is fully connected
    if n_edges != n_nodes*(n_nodes-1)/2:
        raise Exception("Provided graph must be fully connected.")

    Gc = G.copy()

    # check if the provided graph is multilayer
    is_multilayer = True
    if not ("layer" in G.vertex_attributes()): # G is not multilayer
        is_multilayer = False
        preserve_layer = False

    node_idx = [v.index for v in Gc.vs]

    if preserve_layer:
        layers = mlgraph.get_layer_names(Gc)
        node_idx_layer = {}
        for layer in layers:
            node_idx_layer[layer] = [v.index for v in Gc.vs.select(layer_eq = layer)]

    n_swaps *= n_edges

    max_attempt = int(n_nodes*n_edges/(n_nodes*(n_nodes-1)))

    for _ in range(n_swaps):
        attempt = 0
        while attempt <= max_attempt:
            attempt += 1

            # select and edge to swap
            a = random.choice(node_idx)
            b = random.choice(node_idx)

            if a == b:
                continue

            # select a pair of nodes different than (a, b) to consider for swap with (a, b)
            if preserve_layer:
                la = Gc.vs[a]["layer"]
                lb = Gc.vs[b]["layer"]
                c = random.choice(node_idx_layer[la])
                d = random.choice(node_idx_layer[lb])
            else:
                c = random.choice(node_idx)
                d = random.choice(node_idx)

            if c == d:
                continue

            if (c==b or c==a) or (d==b or d==a):
                continue

            # rewire
            edge_ids = Gc.get_eids(pairs = [(a, b), (c, d), (c, b), (a, d)])

            weights = [e["weight"] for e in Gc.es(edge_ids)]

            u = random.uniform(-min(weights[0], weights[1]), min(weights[2], weights[3]))

            Gc.es[edge_ids[0]]["weight"] = weights[0] + u
            Gc.es[edge_ids[2]]["weight"] = weights[2] - u

            Gc.es[edge_ids[1]]["weight"] = weights[1] + u
            Gc.es[edge_ids[3]]["weight"] = weights[3] - u

            break

    return Gc