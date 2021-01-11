# driver_dbscansr.py

import create_data_cluster_sklearn
import dbscan
import matplotlib.pyplot as plt
import metrics
import numpy as np
import plot_data
import time

# (1) generate data
np.random.seed(31)
nsample = 1500
case = "blobs"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
minpts = 5
epsilon = 0.18
model = dbscan.dbscan(minpts,epsilon)
# (3) fit model
model.fit(X)
silhouette = metrics.silhouette(X,model.clustersave[-1])
print("Silhouette Score: {}".format(silhouette))
# (4) plot results
# plot initial data
plot_data.plot_data2d(X,title="Cluster Data")
# plot final clusters
model.plot_cluster()
# plot animation
#model.plot_animation()
plt.show()