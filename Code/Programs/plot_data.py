# plot_data.py

from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
import pandas as pd

def plot_scatter(X,title="",xlabel="",ylabel=""):
    # create scatter plot of data in X (2d numpy array ndim x nsample)
    plt.figure()
    plt.scatter(X[0,:],X[1,:],color=cm.jet(0),marker="o",s=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def plot_scatter_class(X,Y,title="",xlabel="",ylabel=""):
    # create scatter plot of data in X (2d numpy array ndim x nsample)
    # colour based on integer values in Y (1d numpy array)
    list_classname = list(set(Y))
    nclass = len(list_classname)
    # create figure
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i,classname in enumerate(list_classname):
        idx = np.where(Y == classname)[0]
        ax.scatter(X[0,idx],X[1,idx],color = cm.hsv((i+1)/nclass), s=15, label = classname)
    ax.legend(loc="upper left")

def plot_cluster_distribution(cluster_assignment, class_label, figsize=(8,4), figrow=1):
    rcParams.update({'figure.autolayout': True})
    # adjust cluster labels (in case label is -1):
    nsample = np.size(cluster_assignment)
    cluster_assignment_adjusted = deepcopy(cluster_assignment)
    # relabel labels originally set as -1
    for count in range(nsample):
        if cluster_assignment[count] == -1:
            cluster_assignment_adjusted[count] = -(count+1)
    # determine set of labels:
    list_cluster_label = list(set(cluster_assignment_adjusted))
    ncluster = len(list_cluster_label)
    print("Number of Clusters: {}".format(ncluster))
    df = pd.DataFrame({'class': class_label,
                        'cluster label': cluster_assignment_adjusted,
                        'cluster': np.ones(len(class_label))})
    counts = df.groupby(['cluster label', 'class']).sum()
    fig = counts.unstack(level=0).plot(kind='bar', subplots=True,
                                        sharey=True, sharex=False,
                                        layout=(figrow,int(ncluster/figrow)), 
                                        figsize=figsize, legend=False)