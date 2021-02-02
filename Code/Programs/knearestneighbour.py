# knearestneighbour.py

import create_data_cluster_sklearn
import numpy as np
import matplotlib.pyplot as plt

# comment out seed line to generate different set of random numbers
np.random.seed(31)
nsample = 500
case = "noisy_moons"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)

# determine k nearest neighbour
minpts = 6
list_dist = []
for i in range(nsample):
    dist = np.sqrt(np.sum(np.square(X[:,[i]]-X),axis=0))
    dist_sort = np.sort(dist)[minpts-2]
    list_dist.append(dist_sort)

# sort all distances
list_dist.sort()
# plot results
plt.figure()
plt.plot(list_dist)
plt.title("Distance to 5th Nearest Neighbour")
plt.ylabel("Distance")
plt.show()