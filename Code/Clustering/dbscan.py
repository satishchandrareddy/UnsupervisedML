# dbscan_sr.py

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class dbscan:
    def __init__(self,minpts,epsilon):
        self.minpts = minpts
        self.epsilon = epsilon

    def initialize_scan(self,X):
        self.X = X
        self.nsample = X.shape[1]
        self.list_idx = np.random.permutation(self.nsample)
        self.list_label = ["undefined" for _ in range(self.nsample)]
        self.clustersave = [[[count] for count in range(self.nsample)]]

    def pairwise_distance(self,Xidx):
        # compute distance between data point at idx and all othr points
        return np.sqrt(np.sum(np.square(Xidx - self.X),axis=0))

    def neighbours(self,Xidx):
        dist = self.pairwise_distance(Xidx)
        #print("dist: {}".format(dist))
        idx = []
        for count in range(np.size(dist)):
        	if dist[count]<= self.epsilon:
        		idx.append(count)
        return idx

    def add_point(self,list_full,list_seed,list_idx):
        # add additional points to list_neighbour from list_idx if not already in list
        for idx in list_idx:
            if idx not in list_full:
                list_full.append(idx)
                list_seed.append(idx)
        return list_full,list_seed

    def update_clusters(self,cluster1,cluster2):
    	list_cluster = deepcopy(self.clustersave[-1])
    	list_cluster[cluster1].extend(list_cluster[cluster2])
    	list_cluster[cluster2] = []
    	#print("list_cluster: {}".format(list_cluster))
    	self.clustersave.append(list_cluster)

    def fit(self,X):
        self.initialize_scan(X)
        for idx in self.list_idx:
            #print("list_label: {}".format(self.list_label))
            # if point is processed already, then pass
            if self.list_label[idx] != "undefined":
                continue
            # determine neighbours
            list_neighbours = self.neighbours(self.X[:,[idx]])
            # determine if core point
            if len(list_neighbours) < self.minpts:
                self.list_label[idx] = "noise"
                continue
            # idx is a core point
            self.list_label[idx] = "core"
            #print("****New Cluster index: {}".format(idx))
            # points to investigate
            list_seed_full = deepcopy(list_neighbours)
            list_seed = deepcopy(list_neighbours)
            while len(list_seed)> 0:
            	idx_check = list_seed[0]
            	list_seed.pop(0)
            	if self.list_label[idx_check] == "noise":
            		self.list_label[idx_check] = "border"
            		self.update_clusters(idx,idx_check)
            	elif self.list_label[idx_check] != "undefined":
            		continue
            	self.list_label[idx_check] = "border"
            	self.update_clusters(idx,idx_check)
            	new_neighbours = self.neighbours(self.X[:,[idx_check]])
            	if len(new_neighbours)>= self.minpts:
                    self.list_label[idx_check] = "core"
                    list_seed_full,list_seed = self.add_point(list_seed_full,list_seed,new_neighbours)

    def plot_cluster(self,nlevel):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        #ax.set_xlabel("Relative Salary")
        #ax.set_ylabel("Relative Purchases")
        ax.set_title("Clusters")
        cluster_list = self.clustersave[nlevel]
        nsample = len(cluster_list)
        for cluster in range(nsample):
            # plot cluster data
            idx = cluster_list[cluster]
            if len(idx)>0:
                if len(idx)==1:
                    color = 0
                elif len(idx)>1:
                    color = 0.1 + 0.9*(cluster/nsample)
                    print("Cluster size: {}  color: {}".format(len(idx),color))
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
            for cluster in range(len(self.clustersave[frame])):
                idx = self.clustersave[frame][cluster]
                if len(idx)==1:
                    idx1.extend(idx)
                elif len(idx)>1:
                    color = 0.1 + 0.9*(cluster/nframe)
                    image.append(plt.scatter(self.X[0,idx],self.X[1,idx],color=cm.jet(color),marker="o"))
            image.append(plt.scatter(self.X[0,idx1],self.X[1,idx1],color=cm.jet(0),marker="o"))
            container.append(image)
        ani = animation.ArtistAnimation(fig,container, repeat_delay=1000, repeat=False, interval=40, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('dbscan.mp4', writer='ffmpeg')
        if notebook:
            return ani