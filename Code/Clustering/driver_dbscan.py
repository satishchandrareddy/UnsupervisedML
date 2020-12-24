# driver_dbscansr.py

import create_data_cluster_sklearn
import dbscan
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
np.random.seed(31)
nsample = 500
case = "noisy_moons"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
minpts = 3
epsilon = 0.2
model = dbscan.dbscan(minpts,epsilon)
# (3) fit model
start = time.time()
model.fit(X)
end = time.time()
print("Training time (DBScan): {}".format(end-start))

# (4) plot results
# plot initial data
plot_data.plot_data2d(X,title="Cluster Data")
# plot final clusters
model.plot_cluster()
# plot animation
model.plot_animation()
plt.show()