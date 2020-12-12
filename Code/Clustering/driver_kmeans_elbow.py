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
# loop ovr number of clusters to find
np.random.seed(31)
nsim = 10
niteration = 20
ncluster_find = 8
array_objective = []
for ncluster in range(ncluster_find):
	mean_objective = 0
	for sim in range(nsim):
		model = kmeans.kmeans(ncluster+1,"kmeans++")
		list_objective = model.fit(X,niteration,verbose=False)
		mean_objective += list_objective[-1]/nsim 
	print("Number of Clusters: {}   Mean Objective: {}".format(ncluster+1,mean_objective))
	array_objective.append(mean_objective)
# plot data
plot_data.plot_data2d(X)
# plot objective as function of clusters
plot_data.plot_objective([count+1 for count in range(0,ncluster_find)],array_objective,
	title="K Means Elbow Analysis",xlabel="Number of Clusters",ylabel="Objective Function")
plt.show()