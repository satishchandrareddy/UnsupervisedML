# driver_mnist.py

import load_mnist
import kmeans
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) Set up data
np.random.seed(31)
nsample = 60000
X,Y = load_mnist.load_mnist(nsample)
# (2) create model
# initialization should be "random" or "kmeans++"
initialization = "kmeans++"
ncluster = 10
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
nepoch = 20
start = time.time()
list_objective = model.fit(X,nepoch)
end = time.time()
print(f'\n Training time (Kmeans): {end - start}')
# (4) plot results
plot_data.plot_objective(list(range(len(list_objective))),
	list_objective,title="K Means",xlabel="Iteration",ylabel="Objective")
model.plot_cluster_distribution(np.squeeze(Y))
plt.show()