# Exercise_06.3.2.py
# Program to perform elbow analysis for K Means
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
# comment out seed line to generate different sets of random numbers
# loop over number of clusters
max_iter = 50
ncluster_find = 8
list_objective = []
for ncluster in range(ncluster_find):
	np.random.seed(31)
	model = kmeans.kmeans(ncluster+1,"random")
	model.fit(X,max_iter,tolerance=1e-5,verbose=False)
	list_objective.append(model.objectivesave[-1])
	print("Number of Clusters: {}  Objective: {}".format(ncluster+1,list_objective[-1]))
# plot data
plot_data.plot_scatter(X,title="Dataset: "+case,xlabel="Feature x0",ylabel="Feature x1")
# plot objective as function of clusters
plt.figure()
plt.plot(list(range(1,ncluster_find+1)),list_objective,'b-',marker="o",markersize=5)
plt.title("K Means Elbow Analysis Dataset: "+case)
plt.xlabel("Number of Clusters")
plt.ylabel("Objective Function")
plt.show()