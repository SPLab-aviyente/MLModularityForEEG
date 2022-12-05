import numpy as np

from scipy import linalg
from sklearn.cluster import KMeans

from .modularity.matrices import coclustering_matrix

def normalized_adj(A):
    degrees = np.sum(A, axis=1)
    degrees_inv = np.zeros_like(degrees)
    degrees_inv[degrees>0] = degrees[degrees>0]**(-0.5)
    return np.diag(degrees_inv)@A@np.diag(degrees_inv)

def run(A, n_comms, alpha=0.5):
    n_layers = len(A)
    n_nodes, _ = A[0].shape

    # Compute the modified Laplacian from the low dimensional embeddings of layers
    L_mod = np.zeros((n_nodes, n_nodes))
    L_all = np.zeros((n_nodes, n_nodes))

    for layer in range(n_layers):
        L = np.eye(n_nodes) - normalized_adj(A[layer])

        e, V = linalg.eigh(L, subset_by_index=[0, n_comms])

        L_mod += V@V.T
        L_all += L

    # Compute the representative subspace
    L = L_all - alpha*L_mod

    # Find communities
    C = L
    for r in range(10):
        _, V = linalg.eigh(C, subset_by_index=[0, n_comms-1])

        V /= linalg.norm(V, axis=1)[..., None] # normalize rows to unit norm
        km = KMeans(n_comms)
    
        clusters = np.concatenate(
            [km.fit_predict(V)[..., None] for _ in range(100)], axis=1
        )
        C = coclustering_matrix(np.array(clusters))/100
        C[np.diag_indices_from(C)] = 0

        if len(np.unique(C)) == 2:
            break

        C = np.eye(n_nodes) - normalized_adj(C)

    return clusters[:, 0]
