# driver_pca.py

import data_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca

# (1) load MNIST data
nsample = 60000
mnist = data_mnist.mnist()
X,_ = mnist.load(nsample)
# plot 25 random digits from dataset
# set seed to be used when randomly picking images to plot
seed = 11
mnist.plot_image(X,seed,np.arange(nsample))
# (2) create model
model = pca.pca()
# (3) perform fitting (svd on X - Xmean)
model.fit(X)
# (4) reconstruct data using specified variance capture
variance_capture = 0.90
Xr = model.data_reconstructed(variance_capture=variance_capture)
# plot reconstructed data using same seed
seed = 11
mnist.plot_image(Xr,seed,np.arange(nsample))
plt.show()