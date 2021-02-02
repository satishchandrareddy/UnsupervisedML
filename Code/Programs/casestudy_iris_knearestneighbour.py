# casestudy_iris_knearestneighbour.py

import load_iris
import numpy as np
import matplotlib.pyplot as plt

# load data
X,Y = load_iris.load_iris()
nsample = X.shape[1]
# determine k nearest neighbour
minpts = 5
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
plt.title("Distance to K Nearest Neighbour")
plt.ylabel("Distance")
plt.show()