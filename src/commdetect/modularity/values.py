import numpy as np

from . import ml_modularity_matrix, coclustering_matrix

def sl_modularity_value(C, A, P, gamma):
    """Calculate modularity value of a set of community structures of a 
    single-layer graph.

    Parameters
    ----------
    C : numpy ndarray
        Indicator matrix of the set of community structures. C[:, i] is the ith 
        community structure in the set.
    A : sparse or dense ndarray
        The adjacency matrix.
    P : dense ndarray
        The null matrix
    gamma : float
        Resolution parameter.

    Returns
    -------
    mods: float or numpy array
        Calculated modularity values. It is float if C has only community 
        structure, else it is an array.
    """

    B = A - gamma*P
    
    # Calculate modularity values
    if np.ndim(C) == 1:
        n_runs = 1
        C = C[..., None]
    else:
        n_runs = C.shape[1]

    modularities = [np.sum(B[coclustering_matrix(C[:, r])]) for r in range(n_runs)]

    return np.array(modularities) if n_runs > 1 else modularities[0]

def ml_modularity_value(C: np.array, A: dict, P: dict, g1: float, g2: float):
    """Calculate modularity values of a set of community structures of multilayer
    graph.

    Parameters
    ----------
    C : numpy ndarray
        Indicator matrix of the set of community structures. C[:, i] is the ith 
        community structure in the set.
    A : dict of dict
        Supra-adjacency of the multilayer graph.
    P : dict of dict
        Supra-null matrix of the multilayer graph.
    g1 : float
        Resolution parameter.
    g2 : float
        Interlayer scale of the multilayer modularity.

    Returns
    -------
    mods: float or numpy array
        Calculated modularity values. It is float if C has only community 
        structure, else it is an array.
    """
    
    # Construct modularity matrix
    B = ml_modularity_matrix(A, P, g1, g2, as_np=True)

    # Calculate modularity values
    if np.ndim(C) == 1:
        n_runs = 1
        C = C[..., None]
    else:
        n_runs = C.shape[1]

    modularities = [np.sum(B*coclustering_matrix(C[:, r])) for r in range(n_runs)]

    return np.array(modularities) if n_runs > 1 else modularities[0]