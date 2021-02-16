# Exercise 11.1.2

import data_iris
import numpy as np
import matplotlib.pyplot as plt

# load data
iris = data_iris.iris()
X,Y = iris.load()
nsample = X.shape[1]
# determine nearest neighbour
minpts = 5
list_dist = []
for i in range(nsample):
    dist = np.sqrt(np.sum(np.square(X[:,[i]]-X),axis=0))
    dist_sort = np.sort(dist)[minpts-1]
    list_dist.append(dist_sort)

# sort all distances
list_dist.sort()
# plot results
plt.figure()
plt.plot(list_dist)
plt.title("Distance to Nearest Neighbour "+str(minpts))
plt.ylabel("Distance")
plt.show()