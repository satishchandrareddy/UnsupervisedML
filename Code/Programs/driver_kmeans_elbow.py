# driver_kmeans.py

import create_data_cluster_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
nsample = 300
case = "varied_blobs2"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# comment out seed line to generate different sets of random numbers
# loop over number of clusters to find
np.random.seed(31)
nsim = 10
max_iter = 20
ncluster_find = 8
array_objective = []
for ncluster in range(ncluster_find):
	mean_objective = 0
	for sim in range(nsim):
		model = kmeans.kmeans(ncluster+1,"kmeans++")
		model.fit(X,max_iter,tolerance=1e-5,verbose=False)
		mean_objective += model.objectivesave[-1]/nsim 
	print("Number of Clusters: {}   Mean Objective: {}".format(ncluster+1,mean_objective))
	array_objective.append(mean_objective)
# plot data
plot_data.plot_data2d(X)
# plot objective as function of clusters
plt.figure()
plt.plot(list(range(1,ncluster_find+1)),array_objective,'b-',marker="o",markersize=5)
plt.title("K Means Elbow Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Objective Function")
plt.show()