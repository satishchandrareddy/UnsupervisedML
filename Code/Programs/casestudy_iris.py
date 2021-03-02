# driver_iris.py

import data_iris
import hierarchical
import matplotlib.pyplot as plt
import metrics

# (1) load data
iris = data_iris.iris()
X, class_label = iris.load()
# (2) create model
model = hierarchical.hierarchical()
# (3) fit model
model.fit(X)
print("Time fit: {}".format(model.time_fit))
# (4) results
final_level = 3
print("Purity: {}".format(metrics.purity(model.clustersave[-final_level],class_label)))
print("Silhouette: {}".format(metrics.silhouette(X,model.clustersave[-final_level])))
metrics.plot_cluster_distribution(model.clustersave[-final_level],class_label)
plt.show()