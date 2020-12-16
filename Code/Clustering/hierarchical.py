# hierarchical.py
# hierarchical clustering

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class hierarchical:
    def __init__(self):
        self.clustersave = []

    def dist_cluster(self,cluster1,cluster2):
        X_cluster1_mean = np.mean(self.X[:,cluster1],axis=1)
        X_cluster2_mean = np.mean(self.X[:,cluster2],axis=1)
        return np.sqrt(np.sum(np.square(X_cluster1_mean-X_cluster2_mean)))

    def find_dist_cluster_min(self,cluster_list):
        ncluster = len(cluster_list)
        cluster1 = None
        cluster2 = None
        cluster_dist = 1e100
        for count1 in range(ncluster-1):
            for count2 in range(count1+1,ncluster):
                if (len(cluster_list[count1])>0) and (len(cluster_list[count2])>0):
                    dist = self.dist_cluster(cluster_list[count1],cluster_list[count2])
                    if dist<cluster_dist:
                        if len(cluster_list[count1])>=len(cluster_list[count2]):
                            cluster1 = count1
                            cluster2 = count2
                        else:
                            cluster1 = count2
                            cluster2 = count1
                        cluster_dist = dist
        return cluster1, cluster2

    def fit(self,X):
        nsample = X.shape[1]
        self.X = X
        # put each data point in its own cluster
        self.clustersave.append([[count] for count in range(nsample)])
        #print("Level: {}  cluster_list: {}".format(0,self.clustersave[-1]))
        # go through process of combining nearests clusters at each level
        for level in range(nsample-1):
            # find shortest distance between clusters
            cluster_list = deepcopy(self.clustersave[-1])
            cluster1,cluster2 = self.find_dist_cluster_min(cluster_list)
            cluster_list[cluster1].extend(cluster_list[cluster2])
            cluster_list[cluster2]=[]
            self.clustersave.append(cluster_list)
            #print("Level: {} cluster_list: {}".format(level+1,cluster_list))

    def plot_cluster(self,ncluster):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        #ax.set_xlabel("Relative Salary")
        #ax.set_ylabel("Relative Purchases")
        ax.set_title("Clusters")
        cluster_list = self.clustersave[-ncluster]
        nsample = len(cluster_list)
        for cluster in range(nsample):
            # plot cluster data
            idx = cluster_list[cluster]
            if len(idx)>0:
                if len(idx)==1:
                    color = 0
                elif len(idx)>1:
                    color = 0.1 + 0.9*(cluster/nsample)
                clusterdata = plt.scatter(self.X[0,idx],self.X[1,idx],color=cm.jet(color),marker="o")

    def plot_animation(self,ncluster_final=1,notebook=False):
        fig,ax = plt.subplots(1,1)
        #ax.set_xlabel("Relative Salary")
        #ax.set_ylabel("Relative Purchases")
        ax.set_title("Evolution of Clusters")
        nframe = len(self.clustersave)
        container = []
        for frame in range(nframe+1-ncluster_final):
            # create idx to collect single clusters
            idx1 = []
            image = []
            for cluster in range(nframe):
                idx = self.clustersave[frame][cluster]
                if len(idx)==1:
                    idx1.extend(idx)
                elif len(idx)>1:
                    color = 0.1 + 0.9*(cluster/nframe)
                    image.append(plt.scatter(self.X[0,idx],self.X[1,idx],color=cm.jet(color),marker="o"))
            image.append(plt.scatter(self.X[0,idx1],self.X[1,idx1],color=cm.jet(0),marker="o"))
            container.append(image)
        ani = animation.ArtistAnimation(fig,container, repeat_delay=5000, repeat = False, interval=100, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('hierarchical.mp4', writer='ffmpeg')
        if notebook:
            return ani
