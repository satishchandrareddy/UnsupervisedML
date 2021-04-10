# casestudy_bbctext.py

import data_bbctext
import kmeans
import pca
import metrics
import numpy as np
import matplotlib.pyplot as plt
import time

# (1) data
nsample = 2225
text = data_bbctext.bbctext()
# if issues loading dataset because of memory constraints on your machine
# set nsample to be less than 2225
X,class_label = text.load(nsample)
# pca
R = X
use_pca = True
if use_pca:
    time_pca_start = time.time()
    variance_capture = 1.00
    model_pca = pca.pca()
    model_pca.fit(X)
    R = model_pca.data_reduced_dimension(variance_capture=variance_capture)
    print("Time pca: {}".format(time.time() - time_pca_start))
# (2) create model
# initialization should be "random" or "kmeans++"
np.random.seed(71)
initialization = "random"
ncluster = 5
model = kmeans.kmeans(ncluster,initialization)
# (3) fit model
max_iter = 50
tolerance = 1e-4
model.fit(R,max_iter,tolerance)
print("Time fit: {}".format(model.time_fit))
# (4) results
level = -1
print("Purity: {}".format(metrics.purity(model.clustersave[level],class_label)))
print("Davies-Bouldin: {}".format(metrics.davies_bouldin(R,model.clustersave[level])))
model.plot_objective(title="K Means",xlabel="Iteration",ylabel="Objective")
metrics.plot_cluster_distribution(model.clustersave[level],class_label,figsize=(8,4))
text.create_wordcloud(X,model.clustersave[level])
plt.show()