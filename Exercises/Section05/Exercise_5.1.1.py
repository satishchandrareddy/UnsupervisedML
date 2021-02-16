#Exercise_5.1.1.py
#Run in folder UnsupervisedML/Code/Programs

import create_data_cluster_sklearn
import numpy as np
import matplotlib.pyplot as plt

nsample = 200
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)

# determine minputs nearest neightbour
minpts = 4
list_dist = []
for i in range(nsample):
	# compute distance between point at idx i and all points
    dist = np.sqrt(np.sum(np.square(X[:,[i]]-X),axis=0))
    # sort and extract and save minpts nearest distance
    dist_sort = np.sort(dist)[minpts-1]
    list_dist.append(dist_sort)
# sort all saved distances and plot results
list_dist.sort()
plt.figure()
plt.plot(list_dist)
plt.title("Distance to Nearest Neighbour " + str(minpts))
plt.ylabel("Distance")
plt.show()