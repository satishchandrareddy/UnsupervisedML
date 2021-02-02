# create_data.py

import matplotlib.pyplot as plt
import numpy as np
import plot_data
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Datasets from sklearn
def create_data_cluster(n_samples,case):
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
	print("Number of dimensions: {} Number of Samples: {}".format(X.shape[1],X.shape[0]))
	scaler = StandardScaler()
	# translate by sample mean and scale by standard deviation
	return scaler.fit_transform(X).T

if __name__ =="__main__":
	nsample = 500
	case = "noisy_moons"
	title = "Dataset " + case
	X = create_data_cluster(nsample,case)
	plot_data.plot_data2d(X,title=title) 
	plt.show()