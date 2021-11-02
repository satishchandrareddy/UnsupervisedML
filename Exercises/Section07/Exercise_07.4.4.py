# Exercise_07.4.4.py
# Driver for spherical variation of the Gaussian Mixture Model
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import gaussianmm_spherical
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
initialization = "kmeans++"
model = gaussianmm_spherical.gaussianmm(ncluster,initialization)
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
ani = model.plot_cluster_animation(nlevel=-1,interval=500,title="Gaussian Mixture Model Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()