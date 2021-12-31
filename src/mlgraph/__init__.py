from scipy import sparse

def get_layer_names(G):
    return list(set(G.vs["layer"]))

def get_supra_adj(G):
    layers = get_layer_names(G)

    A = G.get_adjacency_sparse(attribute="weight") # get regular adjacency matrix

    node_layer_indices = {} # adjacency matrix indices of nodes of each layer
    for layer in layers:
        node_layer_indices[layer] = [v.index for v in G.vs.select(layer_eq = layer)]

    # construct the supra-adjacency from the regular adjacency matrix
    supra_adj = {}
    for layer1 in layers:
        supra_adj[layer1] = {}

        for layer2 in layers:
            supra_adj[layer1][layer2] = A[node_layer_indices[layer1], :][:, node_layer_indices[layer2]]

    return supra_adj

def get_supra_adj_as_mat(G, layer_order=None):
    supra_adj = get_supra_adj(G)

    # use the layer ordering in supra_adj
    if layer_order is None:
        layer_order = list(supra_adj.keys())

    n_layers = len(layer_order)

    # convert supra adjacency from dict of dict to list of list
    A = []
    for i, layeri in enumerate(layer_order):
        A.append([None]*n_layers)

        for j, layerj in enumerate(layer_order):
            A[i][j] = supra_adj[layeri][layerj]

            print(A[i][j].shape)

    return sparse.bmat(A)

from . import read