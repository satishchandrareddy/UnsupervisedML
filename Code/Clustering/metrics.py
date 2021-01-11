# metrics.py
# cluster metrics functions

from copy import deepcopy
import numpy as np

def silhouette(X,cluster_assignment):
    #Calculate the silhouette value for the dataset after cluster assignments have been determined
    # X is dataset (2d numpy array dimensions: d features x nsamples)
    # cluster_assignment (1d numpy array of length nsamples)
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1
    for count in range(nsample):
        if cluster_assignment[count] == -1:
            cluster_assignment_adjusted[count] = -(count+1)
    # determine set of labels:
    list_cluster_label = list(set(cluster_assignment_adjusted))
    # initialize s
    s = np.zeros((nsample))
    # loop over over labels
    for label in list_cluster_label:
        # determine indices for current current cluster label
        idx_cluster = np.where(np.absolute(cluster_assignment_adjusted - label)<=1e-5)[0]
        if np.size(idx_cluster)==1:
            continue
        # loop over points in the cluster
        for idx in idx_cluster:
            a = np.sum(dist(X[:,[idx]],X[:,idx_cluster]))/(np.size(idx_cluster)-1)
            dist_mean_other = []
            # loop over other clusters to determine b
            for other_label in list_cluster_label:
                if other_label != label:
                    idx_cluster_other = np.where(np.absolute(cluster_assignment_adjusted - other_label)<=1e-5)[0]
                    dist_mean_other.append(np.mean(dist(X[:,[idx]],X[:,idx_cluster_other])))
            b = min(dist_mean_other)
            s[idx] = (b-a)/np.maximum(a,b)
    return np.mean(s)
    
def dist(Xpoint,Xdataset):
    # compute average distance between Xpoint and data in Xdataset
    return np.sqrt(np.sum(np.square(Xpoint - Xdataset),axis=0,keepdims=True))
