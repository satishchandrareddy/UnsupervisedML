# load_mnist.py

import numpy as np
import pandas as pd
from pathlib import Path

def load_mnist(ntrain):
	# read data from train files
	root_dir = Path(__file__).resolve().parent.parent
	dftrain1 = pd.read_csv(root_dir / "Data_MNIST/MNIST_train_set1_30K.csv")
	dftrain2 = pd.read_csv(root_dir / "Data_MNIST/MNIST_train_set2_30K.csv")
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

if __name__ == "__main__":
	Xtrain,Ytrain = load_mnist(6000)