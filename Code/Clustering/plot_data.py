# plot_data.py

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

def plot_objective(list_x,list_y,title="",xlabel="",ylabel=""):
        fig = plt.subplots(1,1)
        plt.plot(list_x,list_y,'b-',marker="o",markersize=5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

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

def plot_data_mnist(X):
    # create 5x5 subplot of mnist images
    nrow = 5
    ncol = 5
    npixel_width = 28
    npixel_height = 28
    fig,ax = plt.subplots(nrow,ncol,sharex="col",sharey="row")
    fig.suptitle("Images of Sample MNIST Digits")
    idx = 0
    for row in range(nrow):
        for col in range(ncol):
            digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
            ax[row,col].pcolormesh(digit_image,cmap="Greys")
            idx +=1
