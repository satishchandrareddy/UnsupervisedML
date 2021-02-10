# driver_pca.py

import data_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca

# (1) load MNIST data
ntrain = 6000
mnist = data_mnist.mnist()
X,_ = mnist.load(ntrain)
# plot 25 random digits from dataset
seed = 11
mnist.plot_image(X,seed)
# (2) create model
model = pca.pca()
# (3) perform fitting (svd on X - Xmean)
model.fit(X)
# (4) reconstruct data using specified variance capture
reduced_dim = 78
variance_capture = 0.90
Xr = model.data_reconstructed(variance_capture=variance_capture)
#Xr = model.data_reconstructed(reduced_dim = reduced_dim)
# plot reconstructed data
mnist.plot_image(Xr,seed)
plt.show()