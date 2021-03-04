# data_mnist.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class mnist:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def load_valid(self,nsample):
    	# load mnist valid dataset
    	# create feature matrix with nsample data points 
	    # read data test
	    df = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_valid_10K.csv")
	    # get labels and 
	    class_label = df["label"]
	    # remove label column
	    df = df.drop(columns="label")
	    # create feature matrix from remaining data - divide by 255 and take transpose
	    X = df.values.T/255
	    # get requested number of data samples
	    X = X[:,:nsample]
	    class_label = class_label[:nsample]
	    print("X.shape: {} - class_label.shape: {}".format(X.shape,class_label.shape))
	    return X,class_label

    def load_train(self,nsample):
    	# load mnist data
    	# create feature matrix with nsample data points 
	    # read data from train files
	    df1 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set1_30K.csv")
	    df2 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set2_30K.csv")
	    # get labels and concatenate
	    class_label1 = df1["label"].values
	    class_label2 = df2["label"].values
	    class_label = np.concatenate((class_label1,class_label2))
	    # remove label column
	    df1 = df1.drop(columns="label")
	    df2 = df2.drop(columns="label")
	    # create feature matrix from remaining data - divide by 255
	    # concatenate and take transpose
	    X1 = df1.values/255
	    X2 = df2.values/255
	    X = np.concatenate((X1,X2),axis=0).T
	    # get requested number of data samples
	    X = X[:,:nsample]
	    class_label = class_label[:nsample]
	    print("X.shape: {} - class_label.shape: {}".format(X.shape,class_label.shape))
	    return X,class_label

    def plot_image(self,X,seed,array_idx):
        # create 5x5 subplot of mnist images
        # X is feature matrix
        # seed is integer used to set up random seed
        # array_idx is numpy array of least length 25
        # choose 25 indices at random from array_idx -> these are images to be plotted
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
            	# convert feature vector (length 784) into 28x28 2darray before plotting
                digit_image = np.flipud(np.reshape(X[:,array_idx_plot[idx]],(npixel_width,npixel_height)))
                ax[row,col].set_aspect("equal")
                ax[row,col].pcolormesh(digit_image,cmap="Greys")
                idx +=1

if __name__ == "__main__":
	# create object
	mnist_object = mnist()
	# load data
	nsample = 60000
	X,_ = mnist_object.load_train(nsample)
	# plot 25 random images from dataset
	seed = 11
	array_idx = np.arange(nsample)
	mnist_object.plot_image(X,seed,array_idx)
	plt.show()