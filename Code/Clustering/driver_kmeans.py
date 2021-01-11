# driver_kmeans.py

import create_data_cluster_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time
import metrics

# (1) generate data
nsample = 1500
ncluster = 3
case = "varied_blobs2"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
# Change seed to change random numbers
# initialization should be "random" or "kmeans++"
np.random.seed(31)
initialization = "kmeans++"
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
niteration = 20
list_objective = model.fit(X,niteration)
silhouette = metrics.silhouette(X,model.clustersave[-1])
print("Silhouette Score: {}".format(silhouette))
# (4) plot results
plot_data.plot_objective(list(range(len(list_objective))),list_objective,
	title="K Means Clustering",xlabel="Iteration",ylabel="Objective")
# plot initial data
plot_data.plot_data2d(X,title="Cluster Data")
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.meansave[0],title="Cluster Data and Initial Means")
# plot final clusters
model.plot_cluster(title="Clustering: Dataset: {}   Silhouette={:.3f}".format(case,silhouette))
# animation
#model.plot_results_animation()
plt.show()