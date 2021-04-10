# casestudy_mnist.py

import data_mnist
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
# set nsample = 10000 or fewer and use valid dataset with line X,class_label = mnist.load_valid(nsample)
X,class_label = mnist.load_train(nsample)
# pca
R = X
use_pca = True
if use_pca:
    time_pca_start = time.time()
    variance_capture = 0.90
    model_pca = pca.pca()
    model_pca.fit(X)
    R = model_pca.data_reduced_dimension(variance_capture=variance_capture)
    print("Time pca: {}".format(time.time() - time_pca_start))
# (2) clustering model
np.random.seed(31)
initialization = "kmeans++"
ncluster = 10
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
max_iter = 100
tolerance = 1e-4
model.fit(R,max_iter,tolerance)
print("Time fit: {}".format(model.time_fit))
# (4) results
level = -1
print("Purity: {}".format(metrics.purity(model.clustersave[level],class_label)))
print("Davies-Bouldin: {}".format(metrics.davies_bouldin(R,model.clustersave[level])))
# plot images from clusters 3,4,5,6
seed = 31
mnist.plot_image(X[:,model.get_index(level,3)],seed)
mnist.plot_image(X[:,model.get_index(level,4)],seed)
mnist.plot_image(X[:,model.get_index(level,5)],seed)
mnist.plot_image(X[:,model.get_index(level,6)],seed)
model.plot_objective(title="K Means Clustering",xlabel="Iteration",ylabel="Objective")
metrics.plot_cluster_distribution(model.clustersave[level],class_label,figsize=(8,4),figrow=2)
plt.show()