# driver_kmeans.py

import create_data_cluster
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(21)
nfeature = 2
nsample = 200
ncluster_data = 3
std = 1
X,mean = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster_data,std)

# loop ovr number of clusters to find
nsim = 10
niteration = 20
ncluster_find = 6
array_objective = []
for ncluster in range(ncluster_find):
	mean_objective = 0
	for sim in range(nsim):
		model = kmeans.kmeans(ncluster+1,"kmeans++")
		mean_objective += model.fit(X,niteration,False)/nsim
	print("Number of Clusters: {}   Mean Objective: {}".format(ncluster+1,mean_objective))
	array_objective.append(mean_objective)

# plot results
array_ncluster = [count+1 for count in range(ncluster_find)]
plot_data.plot_data2d(X)
plt.figure()
plt.plot(array_ncluster,array_objective,marker="o",markersize=5)
plt.xlabel("Number of clusters")
plt.ylabel("Objective Function")
plt.title("K Means Elbow Analysis")
plt.show()