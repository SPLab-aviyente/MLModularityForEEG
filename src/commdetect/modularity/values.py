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

from . import ml_modularity_matrix, coclustering_matrix

def sl_modularity_value(C, A, P, gamma):
    """Calculate modularity values of a set of community structures for a 
    single-layer network.

    Parameters
    ----------
    C : ndarray
        An nxd matrix whose columns is a different community structure, where
        n is the number of nodes, and d is the number of different community 
        structures that are used to construct co-clustering matrix. C[i, j] is 
        the community node i belongs to in jth community structure.
    A : ndarray, or sparse matrix
        Adjacency matrix of the single-layer graph.
    P : ndarray
        Null matrix of the single-layer graph.
    gamma : float
        Resolution parameter.

    Returns
    -------
    modularities: ndarray
        d dimensional vector of modularity values. modularities[i] is the 
        modularity value of C[:, i].
    """

    B = A - gamma*P
    
    # Calculate modularity values
    if np.ndim(C) == 1:
        n_runs = 1
    else:
        n_runs = C.shape[1]

    modularities = [np.sum(B[coclustering_matrix(C[:, r])]) for r in range(n_runs)]

    return np.array(modularities) if n_runs > 1 else modularities[0]

def ml_modularity_value(C: np.array, A: dict, P: dict, g: float, o: float) -> np.array:
    """Calculate modularity values of a set of community structures for a 
    multilayer network.

    Parameters
    ----------
    C : ndarray
        An nxd matrix whose columns is a different community structure, where
        n is the number of nodes, and d is the number of different community 
        structures that are used to construct co-clustering matrix. C[i, j] is 
        the community node i belongs to in jth community structure. Make sure 
        that node order in C is the same as that of given supra-adjacency matrix.
    A : dict of dict
        Supra-adjacency matrix of multilayer network as a dict of dict.
    P : dict of dict
        Supra-null matrix of multilayer network as a dict of dict.
    g : float
        Resolution parameter.
    o : float
        Interlayer scale.

    Returns
    -------
    modularities: ndarray
        d dimensional vector of modularity values. modularities[i] is the 
        modularity value of C[:, i].
    """
    
    # Construct modularity matrix
    B = ml_modularity_matrix(A, P, g, o, as_np=True)

    # Calculate modularity values
    if np.ndim(C) == 1:
        n_runs = 1
        C = C[..., None]
    else:
        n_runs = C.shape[1]

    modularities = [np.sum(B*coclustering_matrix(C[:, r])) for r in range(n_runs)]

    return np.array(modularities) if n_runs > 1 else modularities[0]