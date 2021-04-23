# metrics.py
# cluster metrics functions

from copy import deepcopy
from matplotlib import rcParams
import numpy as np
import pandas as pd

def dist(Xpoint,Xdataset):
    # compute distance between Xpoint and and each point in Xdataset
    # output is 2d array 1 x # of data points
    return np.sqrt(np.sum(np.square(Xpoint - Xdataset),axis=0,keepdims=True))

def davies_bouldin(X,cluster_assignment):
    # Calculate the davies_bouldin index value for dataset
    # X is dataset (2d numpy array: d features x nsample)
    # cluster_assignment (1d numpy array of length nsample)
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1 to -(index + 1)
    for idx in range(nsample):
        if cluster_assignment[idx] == -1:
            cluster_assignment_adjusted[idx] = -(idx+1)
    # determine unique cluster labels
    array_cluster_label = np.unique(cluster_assignment_adjusted)
    ncluster = np.size(array_cluster_label)
    # compute cluster centres and average distance to cluster centre
    cluster_centre = []
    avgdistance_cluster_centre = []
    for cluster in array_cluster_label:
        # indices of points in cluster
        idx_cluster = np.where(np.absolute(cluster_assignment_adjusted - cluster)<1e-5)[0]
        centre = np.mean(X[:,idx_cluster],axis=1,keepdims=True)
        cluster_centre.append(centre)
        avgdistance_cluster_centre.append(np.mean(dist(centre,X[:,idx_cluster])))
    # compute upper triangle of d matrix
    dmat = np.zeros((ncluster,ncluster))
    for i in range(ncluster):
        for j in range(i+1,ncluster):
            dmat[i,j] = avgdistance_cluster_centre[i]+avgdistance_cluster_centre[j]
            dmat[i,j] = dmat[i,j]/dist(cluster_centre[i],cluster_centre[j])
    dmat = dmat + dmat.T
    db = np.mean(np.max(dmat,axis=1))
    return db
    
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
    # determine unique cluster labels
    array_cluster_label = np.unique(cluster_assignment_adjusted)
    # initialize s
    s = np.zeros((nsample))
    # loop over over labels
    for label in array_cluster_label:
        # determine indices for current current cluster label
        idx_cluster = np.where(np.absolute(cluster_assignment_adjusted - label)<=1e-5)[0]
        if np.size(idx_cluster)==1:
            continue
        # loop over points in the cluster
        for idx in idx_cluster:
            a = np.sum(dist(X[:,[idx]],X[:,idx_cluster]))/(np.size(idx_cluster)-1)
            dist_mean_other = []
            # loop over other clusters to determine b
            for other_label in array_cluster_label:
                if other_label != label:
                    # determine indices of points in other cluster and mean distance 
                    idx_cluster_other = np.where(np.absolute(cluster_assignment_adjusted - other_label)<=1e-5)[0]
                    dist_mean_other.append(np.mean(dist(X[:,[idx]],X[:,idx_cluster_other])))
            # determine min of mean distance over all other clusters
            b = min(dist_mean_other)
            s[idx] = (b-a)/np.maximum(a,b)
    return np.mean(s)
    
def purity(cluster_assignment,class_assignment):
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1 to -(index + 1)
    for count in range(nsample):
        if cluster_assignment[count] == -1:
            cluster_assignment_adjusted[count] = -(count+1)
    # determine unique cluster labels
    array_cluster_label = np.unique(cluster_assignment_adjusted)
    total_number = 0
    for cluster in array_cluster_label:
        # determine indices for cluster label
        idx_cluster = np.where(np.absolute(cluster_assignment_adjusted - cluster)<1e-5)[0]
        # count number of times each class appears in cluster
        _,count = np.unique(class_assignment[idx_cluster],return_counts=True)
        # add maximum number to total number
        total_number += np.max(count)
    return total_number/nsample

def plot_cluster_distribution(cluster_assignment, class_assignment, figsize=(8,4), figrow=1):
    rcParams.update({'figure.autolayout': True})
    nsample = np.size(cluster_assignment)
    # determine number of labels
    ncluster = np.size(np.unique(cluster_assignment))
    print("Number of Clusters: {}".format(ncluster))
    # create dataframe
    df = pd.DataFrame({'clusterlabel': cluster_assignment,
                        'classlabel': class_assignment,
                        'cluster': np.ones(np.size(class_assignment))})
    # sum overt clusterlabel and classlabel
    counts = df.groupby(['clusterlabel', 'classlabel']).sum()
    # create bar charts in figrow rows
    fig = counts.unstack(level=0).plot(kind='bar', subplots=True,
                                        sharey=True, sharex=False,
                                        layout=(figrow,int(ncluster/figrow)), 
                                        figsize=figsize, legend=False)