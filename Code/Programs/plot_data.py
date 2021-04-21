# plot_data.py

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_scatter(X,title="",xlabel="",ylabel=""):
    # create scatter plot of data in X (2d numpy array ndim x nsample)
    plt.figure()
    # scatter plot of x0-x1 coordinates of dataset in single color
    plt.scatter(X[0,:],X[1,:],color=cm.jet(0),marker="o",s=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def plot_scatter_class(X,class_label,title="",xlabel="",ylabel=""):
    # create scatter plot of data in X (2d numpy array ndim x nsample)
    # color data points according to labels in class_label (1d numpy array)
    list_class = np.unique(class_label)
    nclass = len(list_class)
    # create figure
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plot x0-x1 coordinates of dataset
    # loop over class labels and color data points with color determined by class label
    for i,classname in enumerate(list_class):
        idx = np.where(class_label == classname)[0]
        ax.scatter(X[0,idx],X[1,idx],color = cm.hsv((i+1)/nclass), s=15, label = str(classname))
    ax.legend(loc="upper left")