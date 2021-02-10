# data_mnist.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class mnist:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def load(self,ntrain):
	    # read data from train files
	    dftrain1 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set1_30K.csv")
	    dftrain2 = pd.read_csv(self.root_dir / "Data_MNIST/MNIST_train_set2_30K.csv")
	    # get labels - concatenate and take transpose to convert to row vector
	    Ytrain1 = dftrain1[["label"]].values
	    Ytrain2 = dftrain2[["label"]].values
	    Ytrain = np.concatenate((Ytrain1,Ytrain2),axis=0).T
	    # remove label column
	    dftrain1 = dftrain1.drop(columns="label")
	    dftrain2 = dftrain2.drop(columns="label")
	    # create feature matrix from remaining data - divide by 255
	    # concatenate and take transpose
	    Xtrain1 = dftrain1.values/255
	    Xtrain2 = dftrain2.values/255
	    Xtrain = np.concatenate((Xtrain1,Xtrain2),axis=0).T
	    # get requested number of training data samples
	    Xtrain = Xtrain[:,:ntrain]
	    Ytrain = Ytrain[:,:ntrain]
	    print("Xtrain.shape: {} - Ytrain.shape: {}".format(Xtrain.shape,Ytrain.shape))
	    return Xtrain,Ytrain

    def plot_image(self,X,seed,array_idx):
        # create 5x5 subplot of mnist images
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
	ntrain = 60000
	Xtrain, _ = mnist_object.load(ntrain)
	# plot 25 random images from dataset
	seed = 11
	mnist_object.plot_image(Xtrain,seed,ntrain)
	plt.show()