# Exercise_09.3.1.py
# Run in UnsupervisedML/Code/Programs

import matplotlib.pyplot as plt
import numpy as np
import pca

# dataset
X = np.array([[1,2,11,-12,2],[3,4,13,1,-7],[-1,-2,-1,-2,-3],[-7,-1,-4,-5,-6]])
print("X: \n{}".format(X))
# create instance of pca class
model = pca.pca()
#(a) compute compact svd and plot of cumulative variance proportion
model.fit(X)
print("Cumulative Variance Proportion: {}".format(model.cumulative_variance_proportion))
model.plot_cumulative_variance_proportion()
#(b) compute reduced dimension version of dataset capturing 0.99 of variance
R = model.data_reduced_dimension(variance_capture=0.99)
print("Reduced dimension R: \n{}".format(R))
#(c) compute reconstructed version of the dataset capturing 0.99 of variance
Xr = model.data_reconstructed(variance_capture=0.99)
print("XR Reconstructed: \n{}".format(Xr))
plt.show()