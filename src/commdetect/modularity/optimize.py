import leidenalg as la
import igraph as ig
import numpy as np

from . import ml_modularity_matrix, bipartite_modularity_matrix, sl_modularity_matrix

def find_sl_communities(A, P, gamma, n_runs=1):
    """Maximize modularity of a single-layer graph using leiden algorithm [1]_.

    Parameters
    ----------
    A : sparse or dense ndarray
        The adjacency matrix.
    P : dense ndarray
        The null matrix
    gamma : float
        Resolution parameter.
    n_runs : int, optional
        Number of times to run leiden algorithm, by default 1

    Returns
    -------
    C : numpy ndarray
        Indicator matrix of the found community structures. Its dimensions are 
        nxr where n is the number of nodes and r is `n_runs`. C[:, i] is the 
        community structure detected at ith run.
    
    References
    ----------
    .. [1] Traag, Vincent A., Ludo Waltman, and Nees Jan Van Eck. "From Louvain
       to Leiden: guaranteeing well-connected communities." Scientific reports 9.1 
       (2019): 1-12.

    """
    B = sl_modularity_matrix(A, P, gamma)

    rng = np.random.default_rng()

    # Find communities
    G = ig.Graph.Weighted_Adjacency(B, mode="undirected", loops=False)
    comms = []
    for r in range(n_runs):
        c = la.find_partition(G, la.CPMVertexPartition, 
                              weights="weight", n_iterations=-1, 
                              resolution_parameter=0,
                              seed=int(rng.uniform()*1e6))
        comms.append(c.membership)
    comms = np.squeeze(np.array(comms)).T

    return comms

def find_bipartite_communities(A, P, gamma, n_runs=1):
    """Maximize modularity of a bipartite graph using leiden algorithm [1]_.

    Parameters
    ----------
    A : sparse or dense ndarray
        The incidence matrix of bipartite graph.
    P : dense ndarray
        The null matrix of bipartite graph generated from its incidence matrix 
        (see `bipartite_null_matrix`)
    gamma : float
        Resolution parameter.
    n_runs : int, optional
        Number of times to run leiden algorithm, by default 1

    Returns
    -------
    C : numpy ndarray
        Indicator matrix of the found community structures. Its dimensions are 
        nxr where n is the number of nodes and r is `n_runs`. C[:, i] is the 
        community structure detected at ith run.

    References
    ----------
    .. [1] Traag, Vincent A., Ludo Waltman, and Nees Jan Van Eck. "From Louvain
       to Leiden: guaranteeing well-connected communities." Scientific reports 9.1 
       (2019): 1-12.
    """

    B = bipartite_modularity_matrix(A, P, gamma)

    rng = np.random.default_rng()

    # Find communities
    G = ig.Graph.Weighted_Adjacency(B, mode="undirected", loops=False)
    comms = []
    for _ in range(n_runs):
        c = la.find_partition(G, la.CPMVertexPartition, weights="weight", 
                              n_iterations=-1, resolution_parameter=0,
                              seed=int(rng.uniform()*1e6))
        comms.append(c.membership)
    comms = np.squeeze(np.array(comms)).T

    return comms

def find_ml_communities(A, P, g1, g2, n_runs=1):
    """Maximize modularity matrix of multilayer graph using leiden algorithm [1]_.

    Parameters
    ----------
    A : dict of dict
        Supra-adjacency of the multilayer graph.
    P : dict of dict
        Supra-null matrix of the multilayer graph.
    g1 : float
        Resolution parameter.
    g2 : float
        Interlayer scale of the multilayer modularity.
    n_runs : int, optional
        Number of times to run leiden algorithm, by default 1

    Returns
    -------
    C : numpy ndarray
        Indicator matrix of the found community structures. Its dimensions are 
        nxr where n is the number of nodes and r is `n_runs`. C[:, i] is the 
        community structure detected at ith run.

    References
    ----------
    .. [1] Traag, Vincent A., Ludo Waltman, and Nees Jan Van Eck. "From Louvain
       to Leiden: guaranteeing well-connected communities." Scientific reports 9.1 
       (2019): 1-12.
    """

    # Get the modularity matrix
    B = ml_modularity_matrix(A, P, g1, g2, as_np=True)

    rng = np.random.default_rng()

    # Find community structures
    G = ig.Graph.Weighted_Adjacency(B, mode="undirected", loops=False)
    comms = []
    for _ in range(n_runs):
        comms.append(
            la.find_partition(G, la.CPMVertexPartition, 
                              weights="weight", n_iterations=-1,
                              resolution_parameter=0, 
                              seed=int(rng.uniform()*1e6)).membership
        )
    comms = np.squeeze(np.array(comms)).T

    return comms