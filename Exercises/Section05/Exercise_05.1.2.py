# Exercise_05.1.2.py
# K nearest neighbour calculation
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import numpy as np
import matplotlib.pyplot as plt

nsample = 200
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)

# determine minpts nearest neightbour
minpts = 4
list_dist = []
for i in range(nsample):
	# compute distance between point at idx i and all points
    dist = np.sqrt(np.sum(np.square(X[:,[i]]-X),axis=0))
    # sort in ascending order and store the minpts nearest neighbour for this data point
    dist_sort = np.sort(dist)[minpts-1]
    list_dist.append(dist_sort)
# sort all nearest neighbour distances in descending order
list_dist.sort(reverse=True)
plt.figure()
plt.plot(list_dist)
plt.title("Distance to Nearest Neighbour " + str(minpts) + "   Data points: " + str(nsample))
plt.ylabel("Distance")
plt.show()