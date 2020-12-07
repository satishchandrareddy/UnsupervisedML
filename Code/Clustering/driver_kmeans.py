# driver_kmeans.py

import create_data_cluster
import create_data_cluster_sklearn
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data
import time

# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(31)
nfeature = 2
nsample = 500
ncluster = 3
std = 1
#X,_ = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster,std)
X = create_data_cluster_sklearn.create_data_cluster(nsample,"varied_blobs")
print("X.shape: {}".format(X.shape))
# (2) create model
# initialization should be "random" or "kmeans++"
initialization = "random"
model = kmeans.kmeans(ncluster,initialization)

# (3) fit model
nepoch = 20
start = time.time()
model.fit(X,nepoch)
end = time.time()
print(f'\n Training time (Kmeans): {end - start}')

# (4) plot results
model.plot_objective()
# plot initial data
plot_data.plot_data2d(X)
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.get_meansave()[0])
# plot final clusters
model.plot_cluster(X)
# animation
model.plot_results_animation(X)
plt.show()