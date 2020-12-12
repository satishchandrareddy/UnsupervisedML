# driver_news_text.py

import load_news_text
import kmeans
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plot_data
rcParams.update({'figure.autolayout': True})

# (1) load data
X, labels = load_news_text.load_news_text()
# (2) create model
# initialization should be "random" or "kmeans++"
np.random.seed(0)
initialization = "kmeans++"
ncluster = 3
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
nepoch = 10
start = time.time()
list_objective = model.fit(X,nepoch)
end = time.time()
print(f'\n Training time (Kmeans): {end - start}')
# (4) plot results
plot_data.plot_objective(list(range(len(list_objective))),list_objective,
					title="K Means",xlabel="Iteration",ylabel="Objective")
model.plot_cluster_distribution(labels, figsize=(12,4))
plt.show()