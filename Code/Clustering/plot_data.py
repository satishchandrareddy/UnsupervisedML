# plot_data.py

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_data2d(X,**kwargs):
    # X is 2d numpy array (nfeature x nsample)
    # kwargs: mean is list of means each (nfeature x 1)
    plt.figure()
    plt.scatter(X[0,:],X[1,:],color=cm.jet(0),marker="o",s=20)
    if "mean" in kwargs:
        mean = kwargs["mean"]
        ncluster = len(mean)
        color_multiplier = 1/ncluster
        for count in range(ncluster):
            color = (count+1)*color_multiplier
            plt.scatter(mean[count][0,0],mean[count][1,0],color=cm.jet(color),marker="s",s=50)
    plt.xlabel("Relative Salary")
    plt.ylabel("Relative Purchases")
    plt.title("Data")

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