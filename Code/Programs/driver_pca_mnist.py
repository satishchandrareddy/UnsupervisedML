# driver_pca.py

import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca
import plot_data

# (1) load MNIST data
ntrain = 6000
X,_ = load_mnist.load_mnist(ntrain)
# plot results - array_image is list of 25 random indices
np.random.seed(11)
array_image = np.random.randint(0,ntrain,25)
plot_data.plot_data_mnist(X,array_image)
# (2) create model
model = pca.pca()
# (3) perform fitting (svd on X - Xmean)
model.fit(X)
# (4) reconstruct data using specified variance capture
reduced_dim = 78
Xr = model.data_reconstructed(reduced_dim = reduced_dim)
# plot reconstructed data
plot_data.plot_data_mnist(Xr,array_image)
plt.show()