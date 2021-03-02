# metrics.py
# cluster metrics functions

from copy import deepcopy
from matplotlib import rcParams
import numpy as np
import pandas as pd

def silhouette(X,cluster_assignment):
    #Calculate the silhouette value for the dataset after cluster assignments have been determined
    # X is dataset (2d numpy array dimensions: d features x nsamples)
    # cluster_assignment (1d numpy array of length nsamples)
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1 to -(index + 1)
    for idx in range(nsample):
        if cluster_assignment[idx] == -1:
            cluster_assignment_adjusted[idx] = -(idx+1)
    # determine list of all cluster labels
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
            # compute a(idx)
            a = np.sum(dist(X[:,[idx]],X[:,idx_cluster]))/(np.size(idx_cluster)-1)
            # find mean distance to other clusters
            mean_dist_other_cluster = []
            for other_label in list_cluster_label:
                if other_label != label:
                    idx_cluster_other = np.where(np.absolute(cluster_assignment_adjusted - other_label)<=1e-5)[0]
                    mean_dist_other_cluster.append(np.sum(dist(X[:,[idx]],X[:,idx_cluster_other]))/np.size(idx_cluster_other))
            # compute b(idx) and s[idx]
            b = min(mean_dist_other_cluster)
            s[idx] = (b-a)/np.maximum(a,b)
    # return silhouette index for dataset 
    return np.mean(s)
    
def dist(Xpoint,Xdataset):
    # compute distance between Xpoint and and each point in Xdataset
    return np.sqrt(np.sum(np.square(Xpoint - Xdataset),axis=0,keepdims=True))

def purity(cluster_assignment,class_assignment):
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1
    for count in range(nsample):
        if cluster_assignment[count] == -1:
            cluster_assignment_adjusted[count] = -(count+1)
    # determine set of labels:
    list_cluster_label = list(set(cluster_assignment_adjusted))
    total_number = 0
    for label in list_cluster_label:
        # determine indices for cluster label
        idx_cluster = np.where(np.absolute(cluster_assignment_adjusted - label)<1e-5)[0]
        # count number of times each class appears in cluster label
        _,count = np.unique(class_assignment[idx_cluster],return_counts=True)
        # add maximum number
        total_number += np.max(count)
    return total_number/nsample

def plot_cluster_distribution(cluster_assignment, class_assignment, figsize=(8,4), figrow=1):
    rcParams.update({'figure.autolayout': True})
    # adjust cluster labels (in case label is -1):
    nsample = np.size(cluster_assignment)
    # determine number of labels
    ncluster = len(set(cluster_assignment))
    print("Number of Clusters: {}".format(ncluster))
    df = pd.DataFrame({'class': class_assignment,
                        'cluster label': cluster_assignment,
                        'cluster': np.ones(len(class_assignment))})
    counts = df.groupby(['cluster label', 'class']).sum()
    fig = counts.unstack(level=0).plot(kind='bar', subplots=True,
                                        sharey=True, sharex=False,
                                        layout=(figrow,int(ncluster/figrow)), 
                                        figsize=figsize, legend=False)