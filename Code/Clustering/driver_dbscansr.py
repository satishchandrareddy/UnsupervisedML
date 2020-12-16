# driver_dbscansr.py

import create_data_cluster_sklearn
import dbscan_sr
import matplotlib.pyplot as plt
import plot_data
import time

# (1) generate data
nsample = 500
case = "noisy_moons"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
minpts = 3
epsilon = 0.2
model = dbscan_sr.dbscan_sr(minpts,epsilon)
# (3) fit model
start = time.time()
model.fit(X)
end = time.time()
print("Training time (DBScan): {}".format(end-start))

# (4) plot results
ncluster = 3
# plot initial data
model.plot_cluster(0)
# plot final clusters
model.plot_cluster(-1)
# plot animation
model.plot_animation(1)
plt.show()