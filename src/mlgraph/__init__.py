import igraph as ig
import numpy as np

from scipy import sparse

from . import io
from . import input_checks

class MLGraph():
    """A composite class that can be used to work with multilayer graphs. The 
    class stores a `igraph` Graph object, while ensuring that nodes have `layer` 
    attribute. 


    Parameters
    ----------
    graph : ig.Graph, optional
        A graph whose nodes have `layer` attribute, by default None. If None, 
        an empty multilayer graph is initiated. Otherwise, the provided graph 
        will be used to initiate the multilayer graph.

    Attributes
    ----------
    graph : ig.Graph
        The underlying ig.Graph. This attribute can be used to access ig.Graph
        methods. 
    layers : list
        List of layer names of the multilayer graph. 
    """

    def __init__(self, graph: ig.Graph=None) -> None:
        
        if graph is None: # Create an empty graph
            self.graph = ig.Graph()
            self.layers = []
        else:
            if not ("layer" in graph.vertex_attributes()):
                raise Exception("Provided graph must have vertex attribute 'layer'.")
            else:
                self.graph = graph
                node_layers = np.array(self.graph.vs["layer"])
                _, indx = np.unique(node_layers, return_index=True)
                self.layers = list(node_layers[np.sort(indx)])

    def order(self, layer=None):
        """Return the number of nodes in a multilayer graph or in a given layer.

        Parameters
        ----------
        layer : str, optional
            Name of the layer whose node number will be returned, by default `None`.

        Returns
        -------
        order: int
            Number of nodes.
        """

        # Input check
        input_checks._layer(layer, self.layers)
        
        # Return number of nodes in the multilayer graph
        if layer is None:
            return self.graph.vcount()

        # Return number of nodes in the given layer
        else:
            return len(self.layer_vertices(layer))

    def size(self, weight=None, layer1=None, layer2=None):
        """Return number of edges in a multilayer graph. If the parameter `weight`
        is not `None`, return total weight of edges.

        Parameters
        ----------
        weight : str, optional
            Edge attribute to use as edge weight, by default `None`.
        layer1 : str, optional
            If not `None` and `layer2` is `None`, consider only intralayer edges
            in `layer1`, by default `None`.
        layer2 : str, optional
            If not `None`, consider only interlayer edges between `layer1` and 
            `layer2`. The parameter `layer1` must be provided, if not `None`.
            By default `None`.

        Returns
        -------
        size: float, or int
            Number of edges or total edge weight if weight is not `None`.
        """
        
        # Input check
        input_checks._edge_attribute(weight, self.graph.edge_attributes())
        input_checks._layer(layer1, self.layers)
        input_checks._layer(layer2, self.layers)
        input_checks._layer_pairs(layer1, layer2)

        # Return the size of multilayer graph
        if layer1 is None and layer2 is None:
            size = sum([1 if weight is None else e["weight"] for e in self.graph.es])

        # Return the size of intralayer graph of layer1
        elif layer2 is None:
            # nodes in the given layer
            layer_nodes = self.layer_vertices(layer1)
            edges = self.graph.es.select(_within = layer_nodes)
            
            size = sum([1 if weight is None else e["weight"] for e in edges])

        # Return the size of interlayer graph between layer1 and layer2
        else:
            # nodes in the given layers
            layer1_nodes = self.layer_vertices(layer1)
            layer2_nodes = self.layer_vertices(layer2)
            edges = self.graph.es.select(_between = (layer1_nodes, layer2_nodes))
            
            size = sum([1 if weight is None else e["weight"] for e in edges])

        return size

    def layer_vertices(self, layer):
        """Return the set of nodes in a given layer.

        Parameters
        ----------
        layer : str
            Layer name.

        Returns
        -------
        nodes: ig.VertexSeq
            Nodes in the given layer.
        """

        input_checks._layer(layer, self.layers)

        return self.graph.vs.select(layer_eq=layer)

    def degree(self, nodes, weight=None, layer=None):
        """Return (weighted, layer-wise) degrees of a set of nodes.

        Parameters
        ----------
        nodes : int, list of ints, or ig.VertexSeq
            A single node ID or a list of node ID.
        weight : str, optional
            Edge attribute to use as weight. If None regular degree will be 
            returned, by default None.
        layer : str, optional
            Name of a layer. If not None, return layer-wise degree of the nodes, 
            by default None.

        Returns
        -------
        degrees: float, or list of floats
            Node degrees.
        """

        input_checks._edge_attribute(weight, self.graph.edge_attributes())
        input_checks._layer(layer, self.layers)

        # If a vertex sequence is provided convert it to the list node ids
        if isinstance(nodes, ig.VertexSeq):
            nodes = nodes.indices

        # If a single vertex is given convert it to a list        
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Return total (weighted) degree of the node
        if layer is None:
            return self.graph.strength(nodes, weights=weight, loops=False)
        
        # Return layer-wise (weighted) degrees of nodes
        else:
            # nodes in the given layer
            layer_nodes = self.layer_vertices(layer).indices

            # Use adjacency matrix to get node strengths
            adj = self.graph.get_adjacency_sparse(attribute=weight)
            return np.sum(adj[nodes, :][:, layer_nodes], axis=1)

    def intralayer_graph(self, layer):
        """Return intralayer graph of a given layer.

        Parameters
        ----------
        layer : str
            The layer whose intralayer graph will be returned.

        Returns
        -------
        G : ig.Graph
            Intralayer graph.
        """
        input_checks._layer(layer, self.layers)

        nodes = self.graph.vs(layer_eq=layer)
        return self.graph.induced_subgraph(nodes)

    def interlayer_graph(self, layer1, layer2):
        """Return interlayer graph that is between two given layers.  

        Parameters
        ----------
        layer1 : str
            Name of the first layer.
        layer2 : str, optional
            Name of the second layer.

        Returns
        -------
        G: ig.Graph
            Intra- or inter-layer graph.
        """

        input_checks._layer(layer1, self.layers)
        input_checks._layer(layer2, self.layers)

        layer1_nodes = self.graph.vs(layer_eq=layer1)
        layer2_nodes = self.graph.vs(layer_eq=layer2)
        subgraph = self.graph.induced_subgraph(
            layer1_nodes.indices + layer2_nodes.indices
        )

        layer1_nodes = subgraph.vs(layer_eq=layer1)
        layer2_nodes = subgraph.vs(layer_eq=layer2)
        edges = subgraph.es(_within=layer1_nodes)
        subgraph.delete_edges(edges)
        edges = subgraph.es(_within=layer2_nodes)
        subgraph.delete_edges(edges)

        return subgraph

    def intralayer_adjacency(self, layer, weight=None):
        """Return adjacency matrix of a given layer.

        Parameters
        ----------
        layer : str
            The layer whose adjacency matrix will be returned.
        weight : str, optional
            Node attribute to use edge weight, by default None. If None, the 
            binary adjacency matrix is returned.

        Returns
        -------
        A : sp.sparse.csr_matrix
            Adjacency matrix. 
        """
        intra_graph = self.intralayer_graph(layer)

        # scipy sparse matrix gives error when graph is empty, 
        # so we handle it by ourselves
        if intra_graph.ecount() == 0:
            n_nodes = intra_graph.vcount()
            return sparse.csr_matrix((n_nodes, n_nodes))
        else:
            return intra_graph.get_adjacency_sparse(attribute=weight)

    def interlayer_incidence(self, layer1, layer2, weight=None):
        """Return incidence matrix of the (bipartite) interlayer graph between 
        two layers.

        Parameters
        ----------
        layer1 : str
            Name of the first layer.
        layer2 : str, optional
            Name of the second layer.
        weight : str, optional
            Node attribute to use edge weight, by default None. If None, the 
            binary incidence matrix is returned.

        Returns
        -------
        A : sp.sparse.csr_matrix
            Incidence matrix. 
        """

        inter_graph = self.interlayer_graph(layer1, layer2)

        # scipy sparse matrix gives error when graph is empty, 
        # so we handle it by ourselves
        if inter_graph.ecount() == 0:
            n_nodes1 = self.order(layer1)
            n_nodes2 = self.order(layer2)
            return sparse.csr_matrix((n_nodes1, n_nodes2))

        inter_graph_adj = inter_graph.get_adjacency_sparse(attribute=weight)

        layer1_nodes = inter_graph.vs(layer_eq=layer1).indices
        layer2_nodes = inter_graph.vs(layer_eq=layer2).indices

        return inter_graph_adj[layer1_nodes, :][:, layer2_nodes]

    def supra_adjacency(self, weight: str=None) -> dict:
        """Return supra-adjacency matrix of the mutlilayer graph.

        Parameters
        ----------
        weight : str, optional
            Node attribute to use edge weight, by default None. If None, the 
            binary supra-adjacency matrix is returned.

        Returns
        -------
        A : dict of dict
            Supra-adjacency matrix as a dict of dict. A[li][lj] is the interlayer
            connectivity between layers li and lj.
        """

        layers = self.layers
        
        supra = {}

        for li in layers:
            supra[li] = {}
            for lj in layers:
                if li == lj:
                    supra[li][li] = self.intralayer_adjacency(li, weight)
                else:
                    supra[li][lj] = self.interlayer_incidence(li, lj, weight)

        return supra
                     
    def read_from_gml(self, file_name: str) -> None:
        """Read the multilayer graph from a gml or gzipped gml file.

        Parameters
        ----------
        file_name : str or Path object.
            File to read. The filename should include the extension: .gml or .gml.gz
        """

        self.graph = io.read_gml(file_name)

        if "layer" not in self.graph.vertex_attributes():
            raise Exception("Graph read is not a multilayer graph. " +\
                            "Make sure vertices have 'layer' attribute.")

        node_layers = np.array(self.graph.vs["layer"])
        _, indx = np.unique(node_layers, return_index=True)
        self.layers = list(node_layers[np.sort(indx)])

    def write_to_gml(self, file_name: str) -> None:
        """Write the multilayer graph to a gml or gzipped 

        Parameters
        ----------
        file_name : str or Path object 
            File to write to. The filename should include the extension: .gml or .gml.gz
        """
        io.write_gml(self.graph, file_name)