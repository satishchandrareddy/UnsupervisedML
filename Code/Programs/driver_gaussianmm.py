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
nsample = 200
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
ncluster = 3
initialization = "random"
model = gaussianmm.gaussianmm(ncluster,initialization)
# (3) fit model
max_iter = 25
tolerance = 1e-3
model.fit(X,max_iter,tolerance)
# (4) plot results
# plot loglikelihood
model.plot_objective(title="Gaussian Mixture Model",xlabel="Iteration",ylabel="Log Likelihood")
# plot initial data with initial means
model.plot_cluster(nlevel=0)
# plot final clusters
model.plot_cluster(nlevel=-1)
# animation
model.plot_cluster_animation(nlevel=-1,interval=400,title="Gaussian Mixture Model",xlabel="Relative Salary",ylabel="Relative Purchases")
plt.show()