# Exercise_07.4.3.py
# Complexity calculation for Gaussian Mixture Model
# Run in folder UnsupervisedML/Code/Programs

import create_dataset_sklearn
import gaussianmm
import matplotlib.pyplot as plt
import numpy as np

# (1) generate data
nsample = 64000
case = "varied_blobs1"
X = create_dataset_sklearn.create_dataset(nsample,case)
array_nsample = np.array([1000,2000,4000,8000,16000,32000,64000])
array_time = np.zeros((np.size(array_nsample)))

# (2) generate time data
# run 5 times to smooth out the data
nrun = 10
for idx in range(np.size(array_nsample)):
    for _ in range(nrun):
        # (2) create model
        ncluster = 3
        model = gaussianmm.gaussianmm(3,"random")
        # (3) fit model
        nsample = array_nsample[idx]
        model.fit(X[:,0:nsample],30,1e-5,False)
        array_time[idx] += model.time_fit
    print("Dimension: {}  Time Fit: {}".format(nsample,array_time[idx]))

# determine power
log_nsample = np.log(array_nsample)
log_time = np.log(array_time)
coeff = np.polyfit(log_nsample,log_time,1)
p = np.poly1d(coeff)
plognsample = p(log_nsample)
print("Power: {}".format(coeff[0]))
plt.figure()
plt.plot(log_nsample,log_time,"ro",label="Data")
plt.plot(log_nsample,plognsample,"b-",label="Fit")
plt.xlabel("Log Dimension")
plt.ylabel("Log Time")
plt.legend()
plt.show()