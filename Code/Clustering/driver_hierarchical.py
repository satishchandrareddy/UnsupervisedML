# driver_hierarchical.py

import create_data_cluster_sklearn
import hierarchical
import matplotlib.pyplot as plt
import plot_data
import time

# (1) generate data
nsample = 205
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
start = time.time()
model.fit(X)
end = time.time()
print("Training time (hierarchical): {}".format(end-start))

# (4) plot results
ncluster = 3
# plot initial data
model.plot_cluster(X.shape[1])
# plot final clusters
model.plot_cluster(ncluster)
# plot animation
model.plot_animation(ncluster)
plt.show()