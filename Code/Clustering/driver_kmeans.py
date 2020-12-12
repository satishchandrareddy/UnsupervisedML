# driver_kmeans.py

import create_data_cluster_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
nsample = 300
ncluster = 3
X = create_data_cluster_sklearn.create_data_cluster(nsample,"varied_blobs2")
# (2) create model
# Change seed to change random numbers
# initialization should be "random" or "kmeans++"
np.random.seed(31)
initialization = "kmeans++"
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
nepoch = 20
start = time.time()
list_objective = model.fit(X,nepoch)
end = time.time()
print("Training time (Kmeans): ".format(end - start))
# (4) plot results
plot_data.plot_objective(list(range(len(list_objective))),list_objective,
	title="K Means Clustering",xlabel="Iteration",ylabel="Objective")
# plot initial data
plot_data.plot_data2d(X,title="Cluster Data")
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.get_meansave()[0],title="Cluster Data and Initial Means")
# plot final clusters
model.plot_cluster(X)
# animation
model.plot_results_animation(X)
plt.show()