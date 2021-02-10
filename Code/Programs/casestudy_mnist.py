# driver_mnist.py

import data_mnist
import gaussianmm
import kmeans
import matplotlib.pyplot as plt
import metrics
import numpy as np
import pca
import plot_data

# (1) set up data
nsample = 6000
mnist = data_mnist.mnist()
X,Y = mnist.load(nsample)
# (2) pca
model_pca = pca.pca()
model_pca.fit(X)
reduced_dim = 87
R = model_pca.data_reduced_dimension(reduced_dim = reduced_dim)
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
print("Purity: {}".format(metrics.purity(model.clustersave[-1],np.squeeze(Y))))
# (4) plot results
seed = 11
mnist.plot_image(X,seed,model.get_index(-1,1))
mnist.plot_image(X,seed,model.get_index(-1,4))
mnist.plot_image(X,seed,model.get_index(-1,5))
mnist.plot_image(X,seed,model.get_index(-1,6))
model.plot_objective(title="Gaussian MM",xlabel="Iteration",ylabel="Objective")
plot_data.plot_cluster_distribution(model.clustersave[-1],np.squeeze(Y),figsize=(8,4),figrow=2)
plt.show()