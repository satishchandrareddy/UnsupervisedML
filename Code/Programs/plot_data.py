# plot_data.py

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

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