import numpy as np

from scipy import sparse

################################################################################
################################ NULL MATRICES #################################
################################################################################

def sl_null_matrix(A, null_model="configuration"):
    """Generate null matrix of single layer modularity from its adjacency matrix.

    Parameters
    ----------
    A : sparse or dense ndarray
        The adjacency matrix of the single layer graph
    null_model : str, optional
        Null model to use to generate null matrix. Could be "configuration" or 
        "erdosrenyi". By default "configuration"

    Returns
    -------
    P : nparray
        Null matrix.

    Raises
    ------
    Exception
        Parameter 'null_model' must be either 'configuration' or 'erdosrenyi'.
    """

    if null_model == "configuration":
        degrees = np.array(np.sum(A, axis=1))
        twom = np.sum(degrees)

        if twom == 0: twom += 0.1 # in case the graph is empty

        null_matrix = np.array(degrees@degrees.T)/twom       
        null_matrix[np.diag_indices_from(null_matrix)] = 0

    elif null_model == "erdosrenyi":
        n_nodes = A.shape[0]
        twom = np.sum(A)
        p = twom/(n_nodes*(n_nodes-1))

        null_matrix = p*np.ones(A.shape)
        null_matrix[np.diag_indices_from(null_matrix)] = 0
    else:
        raise Exception("Parameter 'null_model' must be either 'configuration'",
                        " or 'erdosrenyi'.")
    
    return null_matrix

def bipartite_null_matrix(B, null_model="configuration"):
    """Generate null matrix of bipartite modularity from incidence matrix of the
    bipartite matrix. See [1] for the definition of bipartite modularity.

    Parameters
    ----------
    B : sparse or dense ndarray
        Incidence matrix of the bipartite graph.
    null_model : str, optional
        Null model to use to generate null matrix. Could be "configuration" or 
        "erdosrenyi". By default "configuration"

    Returns
    -------
    P : nparray
        Null matrix.

    Raises
    ------
    Exception
        Parameter 'null_model' must be either 'configuration' or 'erdosrenyi'.

    References
    ---------
    .. [1] Barber, Michael J. "Modularity and community detection in bipartite 
           networks." Physical Review E 76.6 (2007): 066102. 
    """

    if null_model == "configuration":
        degrees_i = np.squeeze(np.array(np.sum(B, axis=1)))[..., None]
        degrees_j = np.squeeze(np.array(np.sum(B, axis=0)))[..., None]
        m = np.sum(degrees_i)

        if m == 0: m += 0.1

        return np.array(degrees_i@degrees_j.T)/m
    elif null_model == "erdosrenyi":
        n_nodes1, n_nodes2 = B.shape
        m = np.sum(B)
        p = m/(n_nodes1*n_nodes2)
        
        return p*np.ones(B.shape)
    else:
        raise Exception("Parameter 'null_model' must be either 'configuration'",
                        " or 'erdosrenyi'.")

def ml_null_matrix(A, layers, null_model="configuration", preserve_layer=True):
    """Generate supra=null matrix of multilayer modularity from supra-adjacency 
    matrix of a multilayer graph.

    Parameters
    ----------
    A : dict of dict
        Supra-adjacency matrix of multilayer graph where A[i][j] is the adjacency
        matrix of ith layer if i==j, and incidence matrix between layers i and j
        if i!=j.
    layers : list
        Name of the layers of the multilayer graph.
    null_model : str, optional
        Null model to use to generate null matrix. Could be "configuration" or 
        "erdosrenyi". By default "configuration"
    preserve_layer : bool, optional
        Whether to preserve the layers when generating supra-null matrix. If True, 
        the supra-null matrix is constructed from null matrix of intra- and inter-layer 
        graphs using single layer (see `sl_null_matrix`) and bipartite modularity
        (see `bipartite_null_matrix`). If false, multilayer graph is considered
        as a single layer graph. By default True

    Returns
    -------
    P : dict of dict
        Supra-null matrix where P[i][j] is the null matrix between layers i and j. 
    """

    if preserve_layer:
        null_matrix = {}
        for li in layers:
            null_matrix[li] = {}
            for lj in layers:
                if li == lj: # Intra-layer modularity matrix
                    null_matrix[li][li] = sl_null_matrix(A[li][li], null_model)
                else:
                    null_matrix[li][lj] = bipartite_null_matrix(A[li][lj], null_model)
    else:
        P = sl_null_matrix(
            sparse.bmat([[A[i][j] for j in A[i]] for i in A]), null_model
        )
        n_nodes = np.cumsum([0] + [A[layers[0]][i].shape[1] for i in A[layers[0]]])
        null_matrix = {}
        for i, li in enumerate(layers):
            null_matrix[li] = {}
            for j, lj in enumerate(layers):
                null_matrix[li][lj] = P[n_nodes[i]:n_nodes[i+1], n_nodes[j]:n_nodes[j+1]]

    return null_matrix

################################################################################
############################# MODULARITY MATRICES ##############################
################################################################################

def sl_modularity_matrix(A, P, gamma):
    """Generate modularity matrix of single layer graph from its adjacency and 
    null matrix.

    Parameters
    ----------
    A : sparse or dense ndarray
        The adjacency matrix.
    P : dense ndarray
        The null matrix
    gamma : float
        Resolution parameter

    Returns
    -------
    B : dense ndarray
        The modularity matrix. 
    """

    return A - gamma*P

def bipartite_modularity_matrix(A, P, gamma):
    """Generate bipartite modularity matrix from its incidence and null matrices.

    Parameters
    ----------
    A : sparse or dense ndarray
        The incidence matrix of bipartite graph.
    P : dense ndarray
        The null matrix of bipartite graph generated from its incidence matrix 
        (see `bipartite_null_matrix`)
    gamma : float
        Resolution parameter.

    Returns
    -------
    B : dense ndarray
        Modularity matrix of the bipartite graph.
    """

    n1, n2 = A.shape # number of nodes in each node type
    n_nodes = n1 + n2
    B = np.zeros((n_nodes, n_nodes))

    B[:n1, n1:] = A - gamma*P
    B[n1:, :n1] = (A - gamma*P).T

    return B

def ml_modularity_matrix(A, P, g1, g2, as_np=False):
    """Generate modularity matrix of a multilayer graph from its supra-adjacency
    and supra-null matrix. 

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
    as_np : bool, optional
        If true, returns modularity matrix as numpy ndarray instead of dict of
        dict. By default False

    Returns
    -------
    B : dict of dict or numpy ndarray
        The modularity matrix of the multilayer graph.
    """
    
    # Init
    B = {layer: {} for layer in A}
    
    # Construct
    for i, layeri in enumerate(A.keys()):
        for j, layerj in enumerate(A.keys()):

            if i==j:
                B[layeri][layerj] = np.array(
                    (A[layeri][layerj] - g1*P[layeri][layerj])
                )
            else:
                B[layeri][layerj] = np.array(
                    g2*(A[layeri][layerj] - g1*P[layeri][layerj])
                )

    if as_np:
        B = np.block(
            [[B[i][j] for j in B[i]] for i in B]
        )
    
    return B

##############
### OTHERS ###
##############

def coclustering_matrix(C):
    """Generate coclustering matrix of a set of community structures.

    Parameters
    ----------
    C : numpy ndarray
        Indicator matrix of the set of community structures. C[:, i] is the ith 
        community structure in the set.

    Returns
    -------
    CC : numpy ndarray
        Coclustering matrix of the input community structures. CC[i][j] is the
        number of times nodes i and j are found in the same communities over 
        input community structures.
    """

    if np.ndim(C) == 1:
        return (C[:, None] == C).astype(int)
    else:
        n_runs = C.shape[1]

        CC = np.array(
            [(C[:, r][:, None] == C[:, r]).astype(int) for r in range(n_runs)]
        )
        return np.squeeze(np.sum(CC, axis=0))

def indicator_matrix(C):
    """Generate indicator matrix for a given community structures.

    Parameters
    ----------
    C : numpy array
        Indicator vector of the community structure, where C[i] is the community
        id of ith node.

    Returns
    -------
    Z : sparse ndarray
        Indicator matrix with dimensions nxk where n is the number of nodes and 
        k is the number of communities. Z[i][r] is 1 if node i in rth community,
        0 otherwise.
    """

    comm_ids, indx = np.unique(C, return_inverse=True)
    n_comms = len(comm_ids)
    n_nodes = len(C)

    if n_comms > 1:
        Z = sparse.csr_matrix(
            (np.ones(n_nodes), (np.arange(n_nodes), indx)), (n_nodes, n_comms)
        )
        return Z
    else:
        return np.ones(n_nodes)