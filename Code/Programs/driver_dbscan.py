# driver_dbscan.py

import create_dataset_sklearn
import dbscan
import matplotlib.pyplot as plt

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
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
# plot final cluster assignments
model.plot_cluster(nlevel=-1, title="DBSCAN Clustering Dataset: "+ case, 
	xlabel="Feature x0", ylabel="Feature x1")
# plot animation
model.plot_cluster_animation(nlevel=-1,interval=100,title="DBSCAN Clustering Dataset: "+case,
	xlabel="Feature x0", ylabel="Feature x1")
plt.show()