# MLModularityForEEG - a python package for modularity based community detection
# in multilayer EEG networks. 
# Copyright (C) 2021 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from scipy import sparse

################################################################################
################################ NULL MATRICES #################################
################################################################################

def sl_null_matrix(A, null_model="configuration"):
    """Construct a null matrix to be used in modularity for single-layer graphs.

    Parameters
    ----------
    A : ndarray, or sparse matrix
        Adjacency matrix of the single-layer graph.
    null_model : str, optional
        Null model to use for constructing null matrix. It can be "configuration"
        or "erdosrenyi". By default "configuration"

    Returns
    -------
    P : ndarray
        Constructed null matrix.
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
    
    return null_matrix

def bipartite_null_matrix(B, null_model="configuration"):
    """Construct a null matrix to be used in modularity for bipartite graphs.

    Parameters
    ----------
    B : ndarray or sparse matrix
        n1xn2 dimensional incidence matrix of the bipartite graph, where n1 and n2
        are number of nodes in the first and second types of the bipartite graph,
        respectively.
    null_model : str, optional
        Null model to use for constructing null matrix. It can be "configuration"
        or "erdosrenyi". By default "configuration"

    Returns
    -------
    P : ndarray
        Constructed null matrix.
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

def ml_null_matrix(A, layers, null_model="configuration", preserve_layer=True):
    """Construct a null matrix to be used in modularity for multilauyer graphs.

    Parameters
    ----------
    A : dict of dict
        Supra-adjacency matrix of multilayer network as a dict of dict.
    layers : list
        Layer names of the multilayer graph.
    null_model : str, optional
        Null model to use for constructing null matrix. It can be "configuration"
        or "erdosrenyi". By default "configuration"
    preserve_layer : bool, optional
        Whether to preserve layers when constructing the null matrix. If true,
        a null matrix is constructed for each intra- and inter-layer graph 
        separately and their concatenation is considered as the null matrix of 
        the multilayer graph. If false, multilayer graph considered as a 
        single-layer graph and null matrix is constructed accordingly. By default 
        True.

    Returns
    -------
    P : dict of dict
        Supra-null matrix of multilayer network as a dict of dict.
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
    """Construct modularity matrix of a single-layer graph.

    Parameters
    ----------
    A : ndarray, or sparse matrix
        Adjacency matrix of the single-layer graph.
    P : ndarray
        Null matrix of the single-layer graph.
    gamma : float
        Resolution parameter.

    Returns
    -------
    B : ndarray
        Modularity matrix.
    """

    return A - gamma*P

def bipartite_modularity_matrix(A, P, gamma):
    """Construct modularity matrix of a bipartite graph.

    Parameters
    ----------
    A : ndarray or sparse matrix
        n1xn2 dimensional incidence matrix of the bipartite graph, where n1 and n2
        are number of nodes in the first and second types of the bipartite graph,
        respectively.
    P : ndarray
        n1xn2 dimensional null matrix of the bipartite graph.
    gamma : float
        Resolution parameter.

    Returns
    -------
    B : ndarray
        nxn dimension modularity matrix, where n=n1+n2.
    """

    n1, n2 = A.shape # number of nodes in each node type
    n_nodes = n1 + n2
    B = np.zeros((n_nodes, n_nodes))

    B[:n1, n1:] = A - gamma*P
    B[n1:, :n1] = (A - gamma*P).T

    return B

def ml_modularity_matrix(A, P, g, o, as_np=False):
    """Construct modularity matrix of a multilayer graph.

    Parameters
    ----------
    A : dict of dict
        Supra-adjacency matrix of multilayer network as a dict of dict.
    P : dict of dict
        Supra-null matrix of multilayer network as a dict of dict.
    g : float
        Resolution parameter.
    o : float
        Interlayer scale.
    as_np : bool, optional
        Whether to return the modularity matrix as a numpy matrix or a dict of 
        dict, by default False.

    Returns
    -------
    B : ndarray, or dict of dict
        Modularity matrix.
    """
    
    # Init
    B = {layer: {} for layer in A}
    
    # Construct
    for i, layeri in enumerate(A.keys()):
        for j, layerj in enumerate(A.keys()):
            if i==j:
                B[layeri][layerj] = np.array(A[layeri][layerj] - g*P[layeri][layerj])
            else:
                B[layeri][layerj] = np.array(o*(A[layeri][layerj] - g*P[layeri][layerj]))

    if as_np:
        B = np.block(
            [[B[i][j] for j in B[i]] for i in B]
        )
    
    return B