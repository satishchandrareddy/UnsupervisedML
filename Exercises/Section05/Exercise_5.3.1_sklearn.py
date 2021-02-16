# Exercise5.3.1.py
# Complexity for DBSCAN

import create_data_cluster_sklearn
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import time

# (1) generate data
nsample = 32000
case = "varied_blobs1"
X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
array_ndim = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
array_time = np.zeros((np.size(array_ndim)))

# (2) generate time data
nrun = 1
for idx in range(np.size(array_ndim)):
    for _ in range(nrun):
        # (2) create model
        minpts = 5
        epsilon = 0.18
        model = DBSCAN(eps=epsilon,min_samples=minpts,animation=False)
        # (3) fit model
        ndim = array_ndim[idx]
        time_start = time.time()
        model.fit(X[:,0:ndim].T)
        array_time[idx] += time.time() - time_start
        #print("Number Neighbours: {}".format(model.nneighbours))
    print("Dimension: {}  Time Fit: {}".format(ndim,array_time[idx]))

# determine power
log_ndim = np.log(array_ndim)
log_time = np.log(array_time)
coeff = np.polyfit(log_ndim,log_time,1)
p = np.poly1d(coeff)
plogndim = p(log_ndim)
print("Power: {}".format(coeff[0]))
plt.figure()
plt.plot(log_ndim,log_time,"ro",label="Data")
plt.plot(log_ndim,plogndim,"b-",label="Fit")
plt.xlabel("Log Dimension")
plt.ylabel("Log Time")
plt.legend()
plt.show()