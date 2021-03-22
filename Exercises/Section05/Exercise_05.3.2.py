# Exercise_05.3.2.py
# Complexity for DBSCAN
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import dbscan
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 32000
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
array_ndim = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
array_time = np.zeros((np.size(array_ndim)))

# (2) generate time data
nrun = 2
for idx in range(np.size(array_ndim)):
    for _ in range(nrun):
        # (2) create model
        minpt = 5
        epsilon = 0.18
        model = dbscan.dbscan(minpt,epsilon,animation=False)
        # (3) fit model
        ndim = array_ndim[idx]
        model.fit(X[:,0:ndim])
        array_time[idx] += model.time_fit
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