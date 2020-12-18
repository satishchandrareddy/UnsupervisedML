# metrics.py

import numpy as np

def silhouette(i, X, cluster_labels, dist_func):
    """Calculate the silhouette value for a given point X_i."""
    # calculate avg distance between X_i and every other point in its cluster
    cluster_label = cluster_labels[i]
    cluster_i = np.squeeze(np.argwhere(cluster_labels == cluster_label))
    C_i = len(cluster_i)
    if C_i == 1:
        return 0
    a_i = np.sum([dist_func(X[:,i], X[:,j]) for j in cluster_i]) / (C_i-1)
    # calculate avg distance between X_i and every other point in its closest cluster
    min_dist = float('inf')
    k = None
    for l in range(X.shape[1]):
        if cluster_labels[l] != cluster_label:
            dist = dist_func(X[:,i], X[:,l])
            if dist < min_dist or k is None:
                min_dist = dist
                k = cluster_labels[l]
    cluster_k = np.squeeze(np.argwhere(cluster_labels == k))
    b_i = np.mean([dist_func(X[:,i], X[:,k]) for k in cluster_k])
    # calculate silhouette value for X_i
    s_i = (b_i - a_i) / max(b_i, a_i)
    return s_i

def silhouette_score(X, cluster_labels, dist_func):
    """Calculate the average silhouette value of all points."""
    S = np.mean([silhouette(i, X, cluster_labels, dist_func) for i in range(X.shape[1])])
    return S