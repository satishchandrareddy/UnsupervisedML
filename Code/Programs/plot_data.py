# plot_data.py

from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

def plot_data2d(X,**kwargs):
    # X is 2d numpy array (nfeature x nsample)
    # kwargs: mean is list of means each (nfeature x 1)
    plt.figure()
    plt.scatter(X[0,:],X[1,:],color=cm.jet(0),marker="o",s=15)
    if "mean" in kwargs:
        list_mean = kwargs["mean"]
        ncluster = len(list_mean)
        color = np.arange(1,ncluster+1)/ncluster
        array_mean = np.concatenate(tuple(list_mean),axis=1)
        plt.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(color),marker="s",s=50)
    if "title" in kwargs:
        plt.title(kwargs["title"])
    if "xlabel" in kwargs:
        plt.xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        plt.ylabel(kwargs["ylabel"])

def plot_scatter_class(X,Y,title="",xlabel="",ylabel=""):
    # determine list of classes:
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

def plot_data_mnist(X,list_image):
    # create 5x5 subplot of mnist images
    nrow = 5
    ncol = 5
    npixel_width = 28
    npixel_height = 28
    fig,ax = plt.subplots(nrow,ncol,sharex="col",sharey="row")
    idx = 0
    fig.suptitle("Images of Sample MNIST Digits")
    for row in range(nrow):
        for col in range(ncol):
            digit_image = np.flipud(np.reshape(X[:,list_image[idx]],(npixel_width,npixel_height)))
            ax[row,col].pcolormesh(digit_image,cmap="Greys")
            idx +=1

def plot_cluster_distribution(cluster_assignment, class_label, figsize=(8,4), figrow=1):
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