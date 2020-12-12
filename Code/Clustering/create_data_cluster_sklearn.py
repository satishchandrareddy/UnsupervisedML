# create_data.py

import matplotlib.pyplot as plt
import numpy as np
import plot_data
from sklearn import datasets

# Anisotropicly distributed data
def create_data_cluster(n_samples,case):
	if case == "aniso":
		random_state = 170
		X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
		transformation = [[0.6, -0.6], [-0.4, 0.8]]
		X = np.dot(X, transformation)
	elif case == "noisy_circles":
		X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
	elif case == "noisy_moons":
		X,y = datasets.make_moons(n_samples=n_samples, noise=.05)
	elif case == "blobs":
		X,y = datasets.make_blobs(n_samples=n_samples, random_state=8)
	elif case == "varied_blobs1":
		random_state = 170
		X,y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
	elif case == "varied_blobs2":
		random_state = 170
		X,y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.5, 2.5, 1.0], random_state=random_state)
	print("Number of dimensions: {} Number of Samples: {}".format(X.shape[1],X.shape[0]))
	return X.T

if __name__ =="__main__":
	nsample = 1500
	case = "varied_blobs1"
	X = create_data_cluster(nsample,case)
	plot_data.plot_data2d(X,title="Data")
	plt.show()