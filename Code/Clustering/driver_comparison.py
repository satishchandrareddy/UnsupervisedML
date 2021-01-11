# driver_comparison.py

import create_data_cluster_sklearn
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import kmeans
import dbscan
import gaussianmm
import metrics

# generate datasets
np.random.seed(31)
nsample = 1500
cases = ["noisy_circles", "noisy_moons", "blobs", "aniso", "varied_blobs1", "varied_blobs2"]
X = {case : create_data_cluster_sklearn.create_data_cluster(nsample, case) for case in cases}

# models:
models = ["K-Means", "GaussianMM", "DBSCAN"]

# pick datasets:
list_dataset = ["blobs", "varied_blobs1", "varied_blobs2"]
#list_dataset = ["noisy_circles", "noisy_moons", "aniso"]
fig, axes = plt.subplots(len(list_dataset), len(models), figsize=(10,15), sharey=True)
for i,dataset in enumerate(list_dataset):
  for j,model in enumerate(models):
    np.random.seed(31)
    if j == 0:
    	axes[i,j].set_ylabel(dataset)
    if dataset == "blobs":
    	if model == "K-Means":
    		mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.3, minpts = 3)
    elif dataset == "varied_blobs1":
    	if model == "K-Means":
    		mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.18, minpts = 5)
    elif dataset == "varied_blobs2":
    	if model == "K-Means":
    		mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.18, minpts = 5)
    elif dataset == "aniso":
    	if model == "K-Means": 
    		mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.18, minpts = 5)
    elif dataset == "noisy_circles":
    	if model == "K-Means": 
    		mod = kmeans.kmeans(ncluster=2, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=2, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.3, minpts = 3)
    elif dataset == "noisy_moons":
    	if model == "K-Means": 
    		mod = kmeans.kmeans(ncluster=2, initialization='kmeans++')
    	elif model == "GaussianMM":
    		mod = gaussianmm.gaussianmm(ncluster=2, initialization='kmeans++')
    	elif model == "DBSCAN":
    		mod = dbscan.dbscan(epsilon=0.3, minpts = 3)
    # fit model
    print("Dataset: {}".format(dataset))
    if model == "DBSCAN":
    	mod.fit(X[dataset])
    else:
    	mod.fit(X[dataset],100,1e-5,False)
    # silhouette score
    silhouette_score = metrics.silhouette(X[dataset],mod.clustersave[-1])
    print("Silhouette Score: {}".format(silhouette_score))
    colors = (mod.clustersave[-1]+1)/mod.ncluster
    axes[i,j].scatter(X[dataset][0,:], X[dataset][1,:], color=cm.jet(colors),s=15)
    axes[i,j].set_xticklabels([])
    axes[i,j].set_yticklabels([])
    if i == 0:
    	title = model + "\ns_score: {:.2f}".format(silhouette_score)
    else:
    	title = "s_score: {:.2f}".format(silhouette_score)
    axes[i,j].set_title(title)
    
plt.show()