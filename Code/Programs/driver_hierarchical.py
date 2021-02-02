# driver_hierarchical.py

import create_data_cluster_sklearn
import hierarchical
import matplotlib.pyplot as plt
import plot_data

# (1) generate data
nsample = 205
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(X)
# (4) plot results
nlevel = 3
# plot initial data
model.plot_cluster(0)
# plot final clusters (at level with 3 clusters)
model.plot_cluster(-nlevel)
# plot animation (to level with 3 clusters)
model.plot_cluster_animation(-nlevel)
plt.show()