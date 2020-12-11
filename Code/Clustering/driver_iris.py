# driver_iris.py

import load_iris
import gaussianmm
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) load data
features, labels = load_iris.load_iris()
ncluster = 3
# (2) create model
initialization = "kmeans++"
model = gaussianmm.gaussianmm(ncluster,initialization)
# (3) fit model
niteration = 20
start = time.time()
list_loglikelihood = model.fit(features,niteration)
end = time.time()
print("Training time (gaussianmm): {}".format(end - start))
# (4) plot results
plot_data.plot_objective(list_loglikelihood,title="Iris",xlabel="Iteration",ylabel="Log Likelihood")
model.plot_cluster_distribution(labels)
plt.show()