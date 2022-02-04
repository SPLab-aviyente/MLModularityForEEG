import numpy as np
import igraph as ig
import leidenalg

from numba import njit
from scipy.spatial.distance import squareform

from .. import mlgraph

# TODO: I need to check if the provided graph is multilayer 
# TODO: A better way to implement this by extending leidenalg?

@njit
def _get_pijs_conf(edgelist, node_layers, lw_strength, layer_sizes):
    pij_intra = np.zeros(edgelist.shape[0])
    pij_inter = np.zeros(edgelist.shape[0])
    for e, edge in enumerate(edgelist):
        i = edge[0]
        j = edge[1]
        li = node_layers[i]
        lj = node_layers[j]
        
        if li == lj:
            pij_intra[e] = (lw_strength[i, lj]*lw_strength[j, li])/(layer_sizes[li, lj]*2)
        else: 
            pij_inter[e] = (lw_strength[i, lj]*lw_strength[j, li])/layer_sizes[li, lj]

    return pij_intra, pij_inter

@njit
def _get_pijs_er(edgelist, node_layers, layer_sizes):
    pij_intra = np.zeros(edgelist.shape[0])
    pij_inter = np.zeros(edgelist.shape[0])

    # get number of nodes in layers
    layer_indices = np.unique(node_layers)
    layer_order = np.zeros(len(layer_indices))
    for i, layer_indx in enumerate(layer_indices):
        layer_order[i] = np.sum(node_layers == layer_indx)

    for e, edge in enumerate(edgelist):
        i = edge[0]
        j = edge[1]
        li = node_layers[i]
        lj = node_layers[j]
        
        if li == lj:
            pij_intra[e] = layer_sizes[li, lj]/layer_order[li]*(layer_order[lj] - 1)/2
        else:
            pij_inter[e] /= layer_sizes[li, lj]/layer_order[li]*layer_order[lj]

    return pij_intra, pij_inter
    
def get_pijs(G, null_model="configuration"):
    # TODO: Docstring

    layers = mlgraph.get_layer_names(G)

    # layer name to layer index
    layers_indx = {layer: i for i, layer in enumerate(layers)}

    edgelist = np.array(np.triu_indices(G.vcount(), k=1)).T
    node_layers = np.array([layers_indx[v["layer"]] for v in G.vs])

    layer_sizes = np.array(
        [[mlgraph.get_layer_sizes(G, l1, l2) for l2 in layers] for l1 in layers]
    )

    if null_model == "configuration":

        # get layerwise strengths of all nodes
        lw_strengths = np.array(
            [[mlgraph.get_layerwise_strength(G, v, l) for l in layers] for v in G.vs]
            )
        
        pij_intra, pij_inter = _get_pijs_conf(edgelist, node_layers, lw_strengths, layer_sizes)

    elif null_model == "erdosrenyi":
        pij_intra, pij_inter = _get_pijs_er(edgelist, node_layers, layer_sizes)
    
    return pij_intra, pij_inter

@njit
def _get_edge_weights(edgelist, edgeweights, node_layers):
    w_intra = np.zeros(edgelist.shape[0])
    w_inter = np.zeros(edgelist.shape[0])
    for e, edge in enumerate(edgelist):
        i = edge[0]
        j = edge[1]
        li = node_layers[i]
        lj = node_layers[j]
        if li == lj:
            w_intra[e] = edgeweights[e]
        else:
            w_inter[e] = edgeweights[e]
    
    return w_intra, w_inter

def get_edge_weights(G):
    # TODO: Binary??
    layers = mlgraph.get_layer_names(G)

    # layer name to layer index
    layers_indx = {layer: i for i, layer in enumerate(layers)}

    edgelist = np.array(np.triu_indices(G.vcount(), k=1)).T
    edgeweights = np.array(G.get_adjacency_sparse(attribute="weight").todense())
    edgeweights = edgeweights[np.triu_indices(G.vcount(), k=1)]
    node_layers = np.array([layers_indx[v["layer"]] for v in G.vs])
    w_intra, w_inter = _get_edge_weights(edgelist, edgeweights, node_layers)

    return w_intra, w_inter

@njit
def _get_B(n_nodes, edgelist, b):
    B = np.zeros((n_nodes, n_nodes))
    for e, edge in enumerate(edgelist):
        B[edge[0], edge[1]] = b[e]
        B[edge[1], edge[0]] = b[e]

    return B

def get_supra_mod(G, null_model="configuration", resolution=1, interlayer_scale=1):
    # TODO: Docstring
    w_intra, w_inter = get_edge_weights(G)
    p_intra, p_inter = get_pijs(G, null_model)

    edgelist = np.array(G.get_edgelist())
    b = (w_intra - resolution*p_intra) - interlayer_scale*(w_inter - resolution*p_inter)
    B = _get_B(G.vcount(), edgelist, b)
    
    layers = mlgraph.get_layer_names(G)

    node_layer_indices = {} # adjacency matrix indices of nodes of each layer
    for layer in layers:
        node_layer_indices[layer] = [v.index for v in G.vs.select(layer_eq = layer)]

    # construct the supra-modularity
    supra_mod = {}
    for layer1 in layers:
        supra_mod[layer1] = {}

        for layer2 in layers:
            supra_mod[layer1][layer2] = B[node_layer_indices[layer1], :][:, node_layer_indices[layer2]]

    return supra_mod

def get_supra_mod_as_mat(G, null_model="configuration", resolution=1, interlayer_scale=1, 
                         layer_order=None):
    # TODO: Docstring
    supra_mod = get_supra_mod(G, null_model, resolution, interlayer_scale)

    # use the layer ordering in supra_adj
    if layer_order is None:
        layer_order = list(supra_mod.keys())

    n_layers = len(layer_order)

    # convert supra adjacency from dict of dict to list of list
    A = []
    for i, layeri in enumerate(layer_order):
        A.append([None]*n_layers)

        for j, layerj in enumerate(layer_order):
            A[i][j] = supra_mod[layeri][layerj]

    return np.block(A)

def find_comms(G, w_intra, w_inter, p_intra, p_inter, resolution=1, interlayer_scale = 1, n_runs = 1):
    # TODO: Docstring
    
    # modularity matrix as a vector
    b = (w_intra - resolution*p_intra) + interlayer_scale*(w_inter - resolution*p_inter)
    edgelist = np.array(np.triu_indices(G.vcount(), k=1)).T
    B = _get_B(G.vcount(), edgelist, b) # modularity matrix

    n = G.vcount()
    rng = np.random.default_rng()

    G_comm = ig.Graph.Full(n)

    partitions = np.zeros((n, n_runs), dtype=int)
    modularities = np.zeros(n_runs)
    for r in range(n_runs):
        c = leidenalg.find_partition(G_comm, leidenalg.CPMVertexPartition, n_iterations=-1, 
                                     weights=b, resolution_parameter=0, 
                                     seed=int(rng.random()*1e4 + r))
        partitions[:, r] = np.array(c.membership, dtype=int)
        modularities[r] = np.sum(B*(partitions[:, r][..., None] == partitions[:, r]).astype(float))

    return partitions, modularities