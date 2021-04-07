# Exercise_09.3.1.py
# Run in UnsupervisedML/Code/Programs

import matplotlib.pyplot as plt
import numpy as np
import pca

# dataset
np.random.seed(11)
X = np.random.randn(50,1000)
# create instance of pca class
model = pca.pca()
#compute compact svd and plot of cumulative variance proportion
model.fit(X)
model.plot_cumulative_variance_proportion()
# compute reconstructed version of dataset with 0.85 proportion of variance
model.data_reconstructed(variance_capture=0.85)
plt.show()