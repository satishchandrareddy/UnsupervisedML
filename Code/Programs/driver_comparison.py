# driver_comparison.py

import create_dataset_sklearn
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import kmeans
import dbscan
import gaussianmm
import metrics

# generate datasets
nsample = 1500
# uncomment one of the following lines
list_dataset = ["blobs", "varied_blobs1", "varied_blobs2"]
#list_dataset = ["noisy_circles", "noisy_moons", "aniso"]
X = {case : create_dataset_sklearn.create_dataset(nsample, case) for case in list_dataset}

# models:
models = ["K Means", "GaussianMM", "DBSCAN"]

# perform clustering and generate plot
fig, axes = plt.subplots(len(list_dataset), len(models), figsize=(8,12), sharey=True)
for i,dataset in enumerate(list_dataset):
    print("Dataset: {}".format(dataset))
    for j,model in enumerate(models):
        np.random.seed(31)
        if j == 0:
    	    axes[i,j].set_ylabel(dataset)
        if dataset == "blobs":
    	    if model == "K Means":
    		    mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        elif dataset == "varied_blobs1":
    	    if model == "K Means":
    		    mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        elif dataset == "varied_blobs2":
    	    if model == "K Means":
    		    mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        elif dataset == "aniso":
    	    if model == "K Means": 
    		    mod = kmeans.kmeans(ncluster=3, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=3, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        elif dataset == "noisy_circles":
    	    if model == "K Means": 
    		    mod = kmeans.kmeans(ncluster=2, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=2, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        elif dataset == "noisy_moons":
    	    if model == "K Means": 
    		    mod = kmeans.kmeans(ncluster=2, initialization='kmeans++')
    	    elif model == "GaussianMM":
    		    mod = gaussianmm.gaussianmm(ncluster=2, initialization='kmeans++')
    	    elif model == "DBSCAN":
    		    mod = dbscan.dbscan(minpts=5, epsilon=0.18)
        # fit model
        print("Model: {}".format(model))
        if model == "DBSCAN":
    	    mod.fit(X[dataset])
        else:
    	    mod.fit(X[dataset],100,1e-5,False)
        print("Time fit: {}".format(mod.time_fit))
        # davies-bouldin and silhouette
        db = metrics.davies_bouldin(X[dataset],mod.clustersave[-1])
        s = metrics.silhouette(X[dataset],mod.clustersave[-1])
        print("Davies-Bouldin: {}".format(db))
        print("Silhouette: {}".format(s))
        colors = (mod.clustersave[-1]+1)/mod.ncluster
        axes[i,j].scatter(X[dataset][0,:], X[dataset][1,:], color=cm.jet(colors),s=15)
        axes[i,j].set_xticklabels([])
        axes[i,j].set_yticklabels([])
        if i == 0:
            title =  model + "\ndb:{:.2f} s:{:.2f} t:{:.3f}".format(db,s,mod.time_fit)
        else:
            title = "db: {:.2f} s:{:.2f} t:{:.3f}".format(db,s,mod.time_fit)
        axes[i,j].set_title(title)
    
plt.show()