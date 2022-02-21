import igraph as ig
import numpy as np
from scipy import sparse


def get_layer_subgraphs(G, layeri, layerj=None):
    if layerj is None: 
        # return a intralayer graph
        nodes = G.vs(layer_eq=layeri)
        return G.induced_subgraph(nodes)
    else: 
        # return an interlayer graph
        layeri_nodes = G.vs(layer_eq=layeri)
        layerj_nodes = G.vs(layer_eq=layerj)
        edges = G.es(_between=(layeri_nodes, layerj_nodes))
        return G.subgraph_edges(edges)

def get_layer_names(G):
    # TODO: Docstring
    return list(set(G.vs["layer"]))

def get_layer_sizes(G, layer1, layer2):
    # TODO: Docstring
    
    if layer1 == layer2:
        layer_nodes = G.vs.select(layer_eq=layer1) # nodes in the given layer
        edges = G.es.select(_within = layer_nodes)
        
        size = sum([e["weight"] for e in edges]) 
    else:
        # nodes in the given layers
        layer1_nodes = G.vs.select(layer_eq=layer1)
        layer2_nodes = G.vs.select(layer_eq=layer2)
        edges = G.es.select(_between = (layer1_nodes, layer2_nodes))
        
        size = sum([e["weight"] for e in edges])

    return size

def get_layerwise_strength(G, node, layer):
    # TODO: Docstring

    # if name of the node or a vertex object is given as node, convert it to node index
    if isinstance(node, ig.Vertex):
        node = node.index
    if isinstance(node, str):
        node = G.vs.find(name=node).index

    layer_nodes = G.vs.select(layer_eq=layer) # nodes in the given layer
    edges = G.es.select(_between=([node], layer_nodes)) # edges between the node and the layer
    strength = sum([e["weight"] for e in edges]) 

    return strength

def get_supra_adj(G, return_node_order=False):
    # TODO: Docstring
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
    # TODO: Docstring
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

    return sparse.bmat(A)

def scale_interlayer(G, w):
    
    layers = get_layer_names(G)
    n_layers = len(layers)

    for i in range(n_layers):
        # Get nodes in ith layer
        li = layers[i]
        li_nodes = G.vs(layer_eq = li)

        for j in range(i+1, n_layers):
            # Get nodes in ith layer
            lj = layers[j]
            lj_nodes = G.vs(layer_eq = lj)

            # Get inter-layer edges between ith and jth layers
            edges = G.es(_between = (li_nodes, lj_nodes))
            
            # Get edge weights
            weights = np.array(edges["weight"])

            # Scale weights
            edges["weight"] = w*weights

from . import read
from . import write