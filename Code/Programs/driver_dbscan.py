# driver_dbscansr.py

import create_data_cluster_sklearn
import dbscan
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 500
case = "noisy_moons"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
minpts = 5
epsilon = 0.18
model = dbscan.dbscan(minpts,epsilon)
# (3) fit model
model.fit(X)
print("Fitting time: {}".format(model.time_fit))
# (4) plot results
# plot initial data
model.plot_cluster(0)
# plot final clusters
model.plot_cluster(-1)
# plot animation
model.plot_cluster_animation(-1)
plt.show()