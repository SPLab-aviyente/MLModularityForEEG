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

def coclustering_matrix(C):
    """Construct co-clustering matrix from a given set of community structures.

    Parameters
    ----------
    C : ndarray
        An nxd matrix whose columns is a different community structure, where
        n is the number of nodes, and d is the number of different community 
        structures that are used to construct co-clustering matrix. C[i, j] is 
        the community node i belongs to in jth community structure.

    Returns
    -------
    CC : ndarray
        nxn co-clustering matrix. C[i, j] is equal to number of times nodes i 
        and j are assigned to the same community in the given set of the 
        community structures.
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
    """Construct an indicator matrix from a given community structure.

    Parameters
    ----------
    C : ndarray
        An n dimensional community assignment vector, where n is number of nodes
        and C[i] is the community node i belongs to. 

    Returns
    -------
    Z : sparse matrix
        nxk dimensional indicator matrix, where k is the number of community in 
        C and Z[i, j] = 1 if node i is in the jth community and 0 otherwise.
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