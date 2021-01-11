# driver_gaussianmm.py

import create_data_cluster_sklearn
import gaussianmm
import matplotlib.pyplot as plt
import metrics
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different set of random numbers
np.random.seed(31)
nsample = 1500
case = "noisy_moons"
ncluster = 2
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
initialization = "kmeans++"
model = gaussianmm.gaussianmm(ncluster,initialization)
# (3) fit model
max_iter = 25
tolerance = 1e-5
list_loglikelihood = model.fit(X,max_iter,tolerance)
silhouette = metrics.silhouette(X,model.clustersave[-1])
print("Silhouette Score: {}".format(silhouette))
# (4) plot results
# plot loglikelihood
plot_data.plot_objective(list(range(len(list_loglikelihood))),list_loglikelihood,
	title="Gaussian Mixture Model",xlabel="Iteration",ylabel="Log Likelihood")
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.meansave[0])
# plot final clusters
model.plot_cluster()
# animation
#model.plot_results_animation()
plt.show()