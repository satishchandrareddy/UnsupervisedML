# driver_hierarchical.py

import create_data_cluster
import create_data_cluster_sklearn
import hierarchical
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different sets of random numbers
nfeature = 2
nsample = 200
ncluster = 3
std = 1
#X,mean = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster,std)
X = create_data_cluster_sklearn.create_data_cluster(nsample,"varied_blobs")
# (2) create model
model = hierarchical.hierarchical()

# (3) fit model
start = time.time()
np.random.seed(20)
model.fit(X)
end = time.time()
print("Training time (hierarchical): {}".format(end-start))

# (4) plot results
# plot initial data
model.plot_cluster(X.shape[1])
# plot final clusters
model.plot_cluster(3)
# plot animation
model.plot_animation(3)
plt.show()