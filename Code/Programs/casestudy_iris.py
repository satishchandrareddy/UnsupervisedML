# driver_iris.py

import data_iris
import hierarchical
import matplotlib.pyplot as plt
import metrics

# (1) load data
iris = data_iris.iris()
X,Y = iris.load()
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(X)
print("Time fit: {}".format(model.time_fit))
# (4) results
print("Purity: {}".format(metrics.purity(model.clustersave[-3],Y)))
print("Silhouette: {}".format(metrics.silhouette(X,model.clustersave[-3])))
metrics.plot_cluster_distribution(model.clustersave[-3],Y)
plt.show()