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
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) load data
X,Y = load_iris.load_iris()
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(X)
# (4) results
print("Purity: {}".format(metrics.purity(model.clustersave[-3],Y)))
print("Silhouette: {}".format(metrics.silhouette(X,model.clustersave[-3])))
model.plot_cluster(-3,"Iris Clustering","Sepal Length","Sepal Width")
plot_data.plot_cluster_distribution(model.clustersave[-3],Y)
plt.show()