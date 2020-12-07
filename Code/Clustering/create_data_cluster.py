# create_data.py

import matplotlib.pyplot as plt
import numpy as np
import plot_data

def create_data_cluster(nfeature,nsample,ncluster,std=1):
	# create ncluster of size roughly nsample/ncluser with points
	# drawn from normal distribution
	npercluster_raw = int(np.ceil(nsample/ncluster))
	npoint = 0
	X = np.zeros((nfeature,0))
	list_mean = []
	for count in range(ncluster):
		list_mean.append(np.random.randn(nfeature,1))
		if count == ncluster-1: #d
			npercluster = nsample - npoint
		else:
			npercluster = npercluster_raw
		X = np.concatenate((X,list_mean[count]+std*np.random.randn(nfeature,npercluster)),axis=1)
		npoint += npercluster
	return X,list_mean
	
if __name__ == "__main__":
	nfeature = 2
	nsample = 200
	ncluster = 4
	std = 0.5
	X,mean = create_data_cluster(nfeature,nsample,ncluster,std)
	print("X.shape: {}".format(X.shape))
	print("mean: {}".format(mean))
	plot_data.plot_data2d(X,mean=mean)
	plt.show()