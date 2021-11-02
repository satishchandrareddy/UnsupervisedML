# driver_kmeans.py

import create_dataset_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
# (2) create model
# Change seed to change random numbers
# initialization should be "random" or "kmeans++"
np.random.seed(31)
ncluster = 3
initialization = "random"
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
max_iter = 30
tolerance = 1e-5
model.fit(X,max_iter,tolerance)
print("Time fit: {}".format(model.time_fit))
# (4) plot results
nlevel = -1
model.plot_objective(title="K Means Clustering Dataset: " +case, xlabel="Iteration", ylabel="Objective")
# plot initial data with initial means
model.plot_cluster(0,title="Initial Means & Datset: "+case, xlabel="Feature x0", ylabel="Feature x1")
# plot final clusters
model.plot_cluster(nlevel=nlevel,title="K Means Clustering Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
# animation
ani = model.plot_cluster_animation(nlevel=-1,interval=800,title="K Means Clustering Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()