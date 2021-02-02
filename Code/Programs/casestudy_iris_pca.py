# driver_iris.py
import dbscan
import load_iris
import gaussianmm
import hierarchical
import kmeans
import matplotlib.pyplot as plt
from matplotlib import rcParams
import metrics
import numpy as np
import pca
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) load data
X,Y = load_iris.load_iris()
# (2) perform pca
use_pca = True
R = X
if use_pca == True:
    model_pca = pca.pca()
    model_pca.fit(X)
    R = model_pca.data_reduced_dimension(reduced_dim=2)
    plot_data.plot_scatter_class(R,Y,"Iris Data Projected to 2 Dimensions using PCA","u0","u1")
# (2) create model
np.random.seed(11)
ncluster = 3
initialization = "kmeans++"
model = gaussianmm.gaussianmm(ncluster,initialization)
# (3) fit model
niteration = 100
model.fit(R,niteration,1e-3)
# (4) results
print("Silhouette: {}".format(metrics.silhouette(R,model.clustersave[-1])))
print("Purity: {}".format(metrics.purity(model.clustersave[-1],Y)))
model.plot_objective(title="Iris",xlabel="Iteration",ylabel="Log Likelihood")
model.plot_cluster(nlevel=-1,title="Iris Clustering using GaussianMM",xlabel="u0",ylabel="u1")
plot_data.plot_cluster_distribution(model.clustersave[-1],Y)
plt.show()