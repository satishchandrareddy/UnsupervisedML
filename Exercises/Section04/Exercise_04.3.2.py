# Exercise_4.3.1.py
# Complexity for hierarchical clustering
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import hierarchical
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 200
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
array_ndim = np.array([10, 20, 50, 100, 200])
array_time = np.zeros((np.size(array_ndim)))

# (2) generate time data
for idx in range(np.size(array_ndim)):
    # (2) create model
    model = hierarchical.hierarchical()
    # (3) fit model
    ndim = array_ndim[idx]
    model.fit(X[:,0:ndim])
    print("Dimension: {}  Time fit: {}".format(ndim,model.time_fit))
    array_time[idx] = model.time_fit

# determine power
log_ndim = np.log(array_ndim)
log_time = np.log(array_time)
coeff = np.polyfit(log_ndim,log_time,1)
p = np.poly1d(coeff)
plogndim = p(log_ndim)
print("Complexity Power: {}".format(coeff[0]))
plt.figure()
plt.plot(log_ndim,log_time,"ro",label="Data")
plt.plot(log_ndim,plogndim,"b-",label="Fit")
plt.xlabel("Log Dimension")
plt.ylabel("Log Time")
plt.legend()
plt.show()