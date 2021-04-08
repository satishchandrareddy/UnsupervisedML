# casestudy_iris_pca.py

import data_iris
import hierarchical
import matplotlib.pyplot as plt
import metrics
import numpy as np
import pca
import plot_data

# (1) load data
iris = data_iris.iris()
X,class_label = iris.load()
# perform pca and reduce dimension to 2
model_pca = pca.pca()
model_pca.fit(X)
R = model_pca.data_reduced_dimension(reduced_dim=2)
plot_data.plot_scatter_class(R,class_label,"Iris Data Projected to 2 Dimensions using PCA","u0","u1")
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(R)
print("Time fit: {}".format(model.time_fit))
# (4) results
level = -3
print("Purity: {}".format(metrics.purity(model.clustersave[level],class_label)))
print("Davies-Bouldin: {}".format(metrics.davies_bouldin(R,model.clustersave[level])))
print("Silhouette: {}".format(metrics.silhouette(R,model.clustersave[level])))
model.plot_cluster(nlevel=level,title="Hierarchical Clustering for Iris Dataset reduced to 2d",xlabel="u0",ylabel="u1")
metrics.plot_cluster_distribution(model.clustersave[level],class_label)
plt.show()