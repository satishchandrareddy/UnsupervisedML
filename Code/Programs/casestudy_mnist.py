# driver_mnist.py

import load_mnist
import kmeans
import gaussianmm
import numpy as np
import metrics
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pca
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) set up data
nsample = 60000
X,Y = load_mnist.load_mnist(nsample)
# (2) pca
model_pca = pca.pca()
model_pca.fit(X)
reduced_dim = 87
R = model_pca.data_reduced_dimension(reduced_dim = reduced_dim)
# (3) clustering model
np.random.seed(31)
initialization = "kmeans++"
ncluster = 10
#model = kmeans.kmeans(ncluster,initialization)
model = gaussianmm.gaussianmm(ncluster,initialization)
# (4) fit model
max_iter = 100
model.fit(R,max_iter)
print("Purity: {}".format(metrics.purity(model.clustersave[-1],np.squeeze(Y))))
# (4) plot results
plot_data.plot_data_mnist(X,model.get_index(model.clustersave[-1],1)[0:25])
plot_data.plot_data_mnist(X,model.get_index(model.clustersave[-1],4)[0:25])
plot_data.plot_data_mnist(X,model.get_index(model.clustersave[-1],5)[0:25])
plot_data.plot_data_mnist(X,model.get_index(model.clustersave[-1],6)[0:25])
model.plot_objective(title="Gaussian MM",xlabel="Iteration",ylabel="Objective")
plot_data.plot_cluster_distribution(model.clustersave[-1],np.squeeze(Y),figsize=(8,4),figrow=2)
plt.show()