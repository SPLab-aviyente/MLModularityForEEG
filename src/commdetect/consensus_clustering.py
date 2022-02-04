import numpy as np
import igraph as ig

def _association_matrix(partitions):
    n_nodes, n_partition = partitions.shape

    A = np.zeros((n_nodes, n_nodes))
    for p in range(n_partition):
        A += (partitions[:, p][..., None] == partitions[:, p]).astype(float)/n_partition

    return A


def find_comms(G, partitions, alg, n_calls=1):
    """Apply consensus clustering to a bunch of comunity structures.

    Consensus clustering finds a consensus community structure from a set of community structures C.  
    First an association graph is constructed where each pair of nodes is connected with an edge 
    whose weight is equal to number of times two nodes are assigned into the same community in C. 
    The association graph is thresholded to remove edges with small weights. Final  association 
    graph is clustered with an algorithm which returns a set of community structures Cn. 
    The consensus clustering is called for Cn. The process continues till community structures in Cn
    are all the same. See [1] for more details. Threshold for association graph is selected as 
    described in [2].

    Parameters
    ----------
    partitions : ndarray
        Community assignments of nodes. partitions[i] is a p dimensional ndarray indicating 
        community assignments of ith node in the set of community structures C, where p = |C|.
    alg : function
        Algorithm to use to cluster association graph. The algorithm should return an ndarray
        indicating community assignments of nodes. 
    Returns
    -------
    C : ndarray
        Consensus community structure. C[i] is the community assignment of node i in the consensus 
        community structure
    References
    ----------
    .. [1] Lancichinetti, Andrea, and Santo Fortunato. "Consensus clustering in complex networks." 
           Scientific reports 2.1 (2012): 1-7.
    .. [2] Bassett, Danielle S., et al. "Robust detection of dynamic community structure in networks." 
           Chaos: An Interdisciplinary Journal of Nonlinear Science 23.1 (2013): 013142.
    """

    # TODO: Input Validation
    
    n_nodes, n_partitions = partitions.shape

    # Original association matrix
    A_org = _association_matrix(partitions)

    # When partitions are the same across runs, association matrix includes only two values
    # We can use this fact to stop the algorithm. 
    if len(np.unique(A_org)) == 2 or n_calls>10:
        return partitions[:, 0]

    # Randomize partitions and get a randomized association matrix
    for p in range(n_partitions):
        np.random.shuffle(partitions[:, p])

    A_rnd = _association_matrix(partitions)

    # Expected number of times a pair of nodes is assigned to the same community in random partition
    threshold = np.max(A_rnd[np.triu_indices(n_nodes, k=1)])

    A_org_c = A_org.copy()
    A_org[A_org<threshold] = 0
    A_org[np.diag_indices(n_nodes)] = 0

    # After thresholding some nodes may be disconnected, connect them to neighbors with high weights  
    degrees = np.count_nonzero(A_org, axis=1)
    for i in np.where(degrees==0)[0]:
        for j in np.where(A_org_c[i, :] >= np.max(A_org_c[i, :])-1e-6)[0]:
            A_org[i, j] = A_org_c[i, j]

    # construct an igraph from association matrix
    T = ig.Graph.Weighted_Adjacency(A_org, mode="undirected")

    T.vs["name"] = G.vs["name"]
    T.vs["layer"] = G.vs["layer"]
    T.vs["electrode"] = G.vs["electrode"]

    return find_comms(G, alg(T), alg, n_calls=n_calls+1)