# driver_gaussianmm.py

import create_data_cluster_sklearn
import gaussianmm
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
# Change seed to change random numbers
# initialization should be "random" or "kmeans++"
np.random.seed(31)
ncluster = 3
initialization = "random"
model = gaussianmm.gaussianmm(ncluster,initialization)
# (3) fit model
max_iter = 30
tolerance = 1e-5
model.fit(X,max_iter,tolerance)
print("Fitting time: {}".format(model.time_fit))
# (4) plot results
# plot loglikelihood
model.plot_objective(title="Gaussian Mixture Model",xlabel="Iteration",ylabel="Log Likelihood")
# plot dataset with initial normal distribution patches
model.plot_cluster(nlevel=0,title="Initial Gaussians & Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
# plot final cluster assignments and distribution pataches
model.plot_cluster(nlevel=-1,title="Gaussian Mixture Model Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
# animation of cluster assignments and distribution patches
model.plot_cluster_animation(nlevel=-1,interval=500,title="Gaussian Mixture Model Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()