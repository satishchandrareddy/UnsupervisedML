# driver_kmeans.py

import create_data_cluster
import create_data_cluster_sklearn
import gaussianmm
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(31)
nsample = 1000
case = "aniso"
ncluster = 3
#X,mean = create_data_cluster_sklearn.create_data_cluster(nsample,case)
X = create_data_cluster_sklearn.create_data_cluster(nsample,"aniso")
# (2) create model
model = gaussianmm.gaussianmm(ncluster)
# (3) fit model
niteration = 20
start = time.time()
model.fit(X,niteration)
end = time.time()
print("Training time (gaussianmm): {}".format(end - start))
# (4) plot results
model.plot_objective()
# plot initial data
plot_data.plot_data2d(X)
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.get_meansave()[0])
# plot final clusters
model.plot_cluster(X)
# animation
#model.plot_results_animation(X)
plt.show()