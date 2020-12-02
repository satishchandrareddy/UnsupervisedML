# driver_hierarchical.py

import create_data_cluster
import hierarchical
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(21)
nfeature = 2
nsample = 200
ncluster = 3
std = 1
X,mean = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster,std)

# (2) create model
model = hierarchical.hierarchical()

# (3) fit model
start = time.time()
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