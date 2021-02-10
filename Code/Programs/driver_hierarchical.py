# driver_hierarchical.py

import create_data_cluster_sklearn
import hierarchical
import matplotlib.pyplot as plt

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(X)
print("Time fit: {}".format(model.time_fit))
# (4) plot results
nlevel = 3
# plot initial data
model.plot_cluster(nlevel=0, title="Data", xlabel="Feature x0", ylabel="Feature x1")
# plot final clusters (at level with 3 clusters)
model.plot_cluster(nlevel=-nlevel, title="Hierarchical Clustering", 
	xlabel="Feature x0", ylabel="Feature x1")
# plot animation (to level with 3 clusters)
model.plot_cluster_animation(nlevel=-nlevel,interval=150,title="Hierarchical Clustering",
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()