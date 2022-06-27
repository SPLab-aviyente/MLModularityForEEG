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

import leidenalg as la
import igraph as ig
import numpy as np

from . import ml_modularity_matrix, bipartite_modularity_matrix, sl_modularity_matrix

def find_sl_communities(A, P, gamma, n_runs=1):
    """Find communities of a single-layer graph by optimizing modularity with
    Leiden algorithm. 

    Parameters
    ----------
    A : ndarray or sparse matrix
        Adjacency matrix of the graph.
    P : ndarray 
        Null matrix of the graph.
    gamma : float
        Resolution parameter.
    n_runs : int, optional
        Number of times to run community detection algorithm, by default 1

    Returns
    -------
    C : ndarray
        nxn_runs dimensional matrix of detected community structures. C[i, j] 
        is the community node i belongs to in the jth run.
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
    """Find communities of a bipartite graph by optimizing modularity with
    Leiden algorithm.

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
    n_runs : int, optional
        Number of times to run community detection algorithm, by default 1

    Returns
    -------
    C : ndarray
        nxn_runs dimensional matrix of detected community structures. C[i, j] 
        is the community node i belongs to in the jth run. n is equal to n1+n2, 
        and first n1 rows C correspond to nodes in first type and remaining ones
        correcpond to second type.
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

def find_ml_communities(A, P, g, o, n_runs=1):
    """Find communities of a multilayer graph by optimizing modularity with
    Leiden algorithm. 

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
    n_runs : int, optional
        Number of times to run community detection algorithm, by default 1

    Returns
    -------
    C : ndarray
        nxn_runs dimensional matrix of detected community structures. C[i, j] 
        is the community node i belongs to in the jth run. Node ordering of C
        is the same as ordering in supra-adjacency.
    """


    # Get the modularity matrix
    B = ml_modularity_matrix(A, P, g, o, as_np=True)

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