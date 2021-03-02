# data_mnist.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class mnist:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def load(self,nsample):
    	# load mnist data
    	# create feature matrix with nsample data points 
	    # read data from train files
	    df1 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set1_30K.csv")
	    df2 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set2_30K.csv")
	    # get labels and concatenate
	    Y1 = df1["label"].values
	    Y2 = df2["label"].values
	    Y = np.concatenate((Y1,Y2))
	    # remove label column
	    df1 = df1.drop(columns="label")
	    df2 = df2.drop(columns="label")
	    # create feature matrix from remaining data - divide by 255
	    # concatenate and take transpose
	    X1 = df1.values/255
	    X2 = df2.values/255
	    X = np.concatenate((X1,X2),axis=0).T
	    # get requested number of training data samples
	    X = X[:,:nsample]
	    Y = Y[:nsample]
	    print("X.shape: {} - Y.shape: {}".format(X.shape,Y.shape))
	    return X,Y

    def plot_image(self,X,seed,array_idx):
        # create 5x5 subplot of mnist images
        # X is feature matrix
        # seed is integer used to set up random seed
        # array_idx is numpy array of least length 25
        # randomly choose 25 indices from array_idx -> these are images to be plotted
        np.random.seed(seed)
        array_idx_plot = np.random.choice(array_idx,25,replace=False)
        nrow = 5
        ncol = 5
        npixel_width = 28
        npixel_height = 28
        fig,ax = plt.subplots(nrow,ncol,sharex="col",sharey="row")
        idx = 0
        fig.suptitle("Images of Sample MNIST Digits")
        for row in range(nrow):
            for col in range(ncol):
                digit_image = np.flipud(np.reshape(X[:,array_idx_plot[idx]],(npixel_width,npixel_height)))
                ax[row,col].pcolormesh(digit_image,cmap="Greys")
                idx +=1

if __name__ == "__main__":
	# create object
	mnist_object = mnist()
	# load data
	nsample = 60000
	X, _ = mnist_object.load(nsample)
	# plot 25 random images from dataset
	seed = 11
	array_idx = np.arange(nsample)
	mnist_object.plot_image(X,seed,array_idx)
	plt.show()