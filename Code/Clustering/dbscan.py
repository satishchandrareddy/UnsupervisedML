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
        self.clustersave = [(-1)*np.ones((self.nsample))]

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

    def update_clusters(self,cluster_number,idx):
    	list_cluster = deepcopy(self.clustersave[-1])
    	list_cluster[idx] = cluster_number
    	self.clustersave.append(list_cluster)

    def fit(self,X):
        self.initialize_scan(X)
        cluster_number = -1
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
            # idx is a core point - create new cluster by updating cluster number
            cluster_number += 1
            self.list_label[idx] = "core"
            self.update_clusters(cluster_number,idx)
            #print("****New Cluster index: {}".format(idx))
            # points to investigate
            list_seed_full = deepcopy(list_neighbours)
            list_seed = deepcopy(list_neighbours)
            while len(list_seed)> 0:
            	idx_check = list_seed[0]
            	list_seed.pop(0)
            	if self.list_label[idx_check] == "noise":
            		self.list_label[idx_check] = "border"
            		self.update_clusters(cluster_number,idx_check)
            	elif self.list_label[idx_check] != "undefined":
            		continue
            	self.list_label[idx_check] = "border"
            	self.update_clusters(cluster_number,idx_check)
            	new_neighbours = self.neighbours(self.X[:,[idx_check]])
            	if len(new_neighbours)>= self.minpts:
                    self.list_label[idx_check] = "core"
                    list_seed_full,list_seed = self.add_point(list_seed_full,list_seed,new_neighbours)
        self.ncluster = cluster_number + 1

    def plot_cluster(self):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title("Clusters")
        array_color_data = (1+self.clustersave[-1])/(self.ncluster+1e-16)
        scatter_data = plt.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)

    def plot_animation(self,notebook=False):
        fig,ax = plt.subplots(1,1)
        ax.set_title("Evolution of Clusters")
        nframe = len(self.clustersave)
        print("nframe: {}".format(nframe))
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)

        def update(i,scat,clustersave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat.set_color(cm.jet(array_color_data))
            return scat,

        ani = animation.FuncAnimation(fig=fig, func=update, frames = len(self.clustersave),
            fargs=[scat,self.clustersave,self.ncluster], repeat_delay=1000, repeat=True, interval=40, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        ani.save('dbscan.mp4', writer='ffmpeg')
        if notebook:
            return ani