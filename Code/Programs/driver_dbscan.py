# driver_dbscansr.py

import create_data_cluster_sklearn
import dbscan
import matplotlib.pyplot as plt

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
# (2) create model
minpts = 4
epsilon = 0.3
model = dbscan.dbscan(minpts,epsilon)
# (3) fit model
model.fit(X)
print("Fitting time: {}".format(model.time_fit))
# (4) plot results
# plot initial data
model.plot_cluster(nlevel=0, title="Dataset: "+case, xlabel="Feature x0", ylabel="Feature x1")
# plot final clusters (at level with 3 clusters)
model.plot_cluster(nlevel=-1, title="DBSCAN Clustering Dataset: "+ case, 
	xlabel="Feature x0", ylabel="Feature x1")
# plot animation (to level with 3 clusters)
model.plot_cluster_animation(nlevel=-1,interval=40,title="DBSCAN Clustering Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()