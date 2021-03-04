# driver_mnist.py

import data_mnist
import gaussianmm
import kmeans
import matplotlib.pyplot as plt
import metrics
import numpy as np
import pca
import time

# (1) set up data
nsample = 60000
mnist = data_mnist.mnist()
# if issues loading dataset because of memory constraints on your machine
# set nsample = 10000 or fewer and use line X,_ = mnist.load_valid(nsample)
X,class_label = mnist.load_train(nsample)
# (2) pca
R = X
use_pca = False
if use_pca:
    time_pca_start = time.time()
    variance_capture = 0.90
    model_pca = pca.pca()
    model_pca.fit(X)
    R = model_pca.data_reduced_dimension(variance_capture=variance_capture)
    print("Time pca: {}".format(time.time() - time_pca_start))
# (3) clustering model
np.random.seed(31)
initialization = "kmeans++"
ncluster = 10
model = kmeans.kmeans(ncluster,initialization)
#model = gaussianmm.gaussianmm(ncluster,initialization)
# (4) fit model
max_iter = 100
model.fit(R,max_iter)
print("Time fit: {}".format(model.time_fit))
print("Purity: {}".format(metrics.purity(model.clustersave[-1],class_label)))
# (5) plot results
seed = 11
mnist.plot_image(X[:,model.get_index(-1,1)],seed)
mnist.plot_image(X[:,model.get_index(-1,4)],seed)
mnist.plot_image(X[:,model.get_index(-1,5)],seed)
mnist.plot_image(X[:,model.get_index(-1,6)],seed)
model.plot_objective(title="Gaussian MM",xlabel="Iteration",ylabel="Objective")
metrics.plot_cluster_distribution(model.clustersave[-1],class_label,figsize=(8,4),figrow=2)
plt.show()