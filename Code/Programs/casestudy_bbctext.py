# driver_news_text.py

import data_bbctext
import kmeans
import pca
import metrics
import numpy as np
import matplotlib.pyplot as plt
import time

# (1) load data
text = data_bbctext.bbctext()
X,Y = text.load()
# (2) PCA
R = X
use_pca = True
if use_pca:
    time_pca_start = time.time()
    variance_capture = 1.00
    model_pca = pca.pca()
    model_pca.fit(X)
    R = model_pca.data_reduced_dimension(variance_capture=variance_capture)
    print("Time pca: {}".format(time.time() - time_pca_start))
# (3) create model
# initialization should be "random" or "kmeans++"
np.random.seed(71)
initialization = "random"
ncluster = 5
model = kmeans.kmeans(ncluster,initialization)
# (4) fit model
max_iter = 50
model.fit(R,max_iter)
print("Time fit: {}".format(model.time_fit))
# (5) results
print("Purity: {}".format(metrics.purity(model.clustersave[-1],np.squeeze(Y))))
model.plot_objective(title="K Means",xlabel="Iteration",ylabel="Objective")
metrics.plot_cluster_distribution(model.clustersave[-1],Y,figsize=(8,4))
text.create_wordcloud(X,model.clustersave[-1],ncluster)
plt.show()