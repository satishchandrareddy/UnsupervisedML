# driver_pca_mnist.py

import data_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca

# (1) load MNIST data
nsample = 60000
mnist = data_mnist.mnist()
# if issues loading dataset because of memory constraints on your machine
# set nsample = 10000 or fewer and use line X,_ = mnist.load_valid(nsample)
X,_ = mnist.load_train(nsample)
# plot 25 random digits from dataset
# set seed to be used when randomly picking images to plot
seed = 11
mnist.plot_image(X,seed)
# (2) create model
model = pca.pca()
# (3) perform fitting (svd on X - Xmean)
model.fit(X)
# plot cumulative_variance_proportion
model.plot_cumulative_variance_proportion()
# (4) reconstruct data using specified variance capture
variance_capture = 0.90
Xr = model.data_reconstructed(variance_capture=variance_capture)
# plot reconstructed data using same seed
mnist.plot_image(Xr,seed)
plt.show()