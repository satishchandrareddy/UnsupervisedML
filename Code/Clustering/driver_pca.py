# driver_pca.py

import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca
import plot_data

# (1) load MNIST data
ntrain = 6000
X,_ = load_mnist.load_mnist(ntrain)
plot_data.plot_data_mnist(X)
# (2) create model
model = pca.pca()
# (3) perform fitting (svd on X - Xmean)
model.fit(X)
# (4) reconstruct data set using principal components to capture proportion of variance
variance_capture = 0.99
Xr = model.data_reconstructed(variance_capture = variance_capture)
# plot reconstructed data
plot_data.plot_data_mnist(Xr)
plt.show()