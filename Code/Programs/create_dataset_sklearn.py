# create_dataset_sklearn.py

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Datasets from sklearn - see also UnsupervisedML/Examples/Section02/SklearnDatasets.ipynb
def create_dataset(n_samples,case):
	if case == "aniso":
		X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
		transformation = [[0.6, -0.6], [-0.4, 0.8]]
		X = np.dot(X, transformation)
	elif case == "noisy_circles":
		X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=170)
	elif case == "noisy_moons":
		X,y = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=170)
	elif case == "blobs":
		X,y = datasets.make_blobs(n_samples=n_samples, random_state=8)
	elif case == "varied_blobs1":
		X,y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5], random_state=170)
	elif case == "varied_blobs2":
		X,y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.5, 3.5, 0.8], random_state=170)
	print("Case: {}  Number of dimensions: {} Number of Samples: {}".format(case,X.shape[1],X.shape[0]))
	# create scaler to translate by sample mean and scale by standard deviation
	scaler = StandardScaler()
	# take transpose so sample axis is column
	return scaler.fit_transform(X).T

if __name__ == "__main__":
    # create each of datasets with 1500 points and plot in 2 rows and 3 columns
    list_dataset = [["blobs", "varied_blobs1", "varied_blobs2"],["aniso", "noisy_moons", "noisy_circles"]]
    nsample = 1500
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow,ncol)
    for row in range(nrow):
	    for col in range(ncol):
		    case = list_dataset[row][col]
		    X = create_dataset(nsample,case)
		    ax[row,col].scatter(X[0,:],X[1,:],color = cm.jet(0), s=15)
		    ax[row,col].set_title(case)
    plt.show()