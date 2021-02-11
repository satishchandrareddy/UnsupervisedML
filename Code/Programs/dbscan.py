# dbscan_sr.py

import clustering_base
from copy import deepcopy
import numpy as np
import time

class dbscan(clustering_base.clustering_base):
    def __init__(self,minpts,epsilon):
        self.minpts = minpts
        self.epsilon = epsilon

    def initialize_algorithm(self):
        self.objectivesave = []
        self.list_label = ["unvisited" for _ in range(self.nsample)]
        self.clustersave = [(-1)*np.ones((self.nsample))]
        self.nneighbours = 0

    def neighbours(self,Xidx):
        dist = np.sqrt(np.sum(np.square(Xidx - self.X),axis=0))
        self.nneighbours += 1
        return list(np.where(dist<=self.epsilon)[0])
    
    def add_points(self,list_seed,list_idx):
        # add additional points to list_seed if unvisited or noise 
        for idx in list_idx:
            if self.list_label[idx] == "unvisited" or self.list_label[idx] == "noise":
                list_seed.append(idx)
        return list(set(list_seed))

    def update_cluster_assignment(self,cluster_number,idx):
    	list_cluster = deepcopy(self.clustersave[-1])
    	list_cluster[idx] = cluster_number
    	self.clustersave.append(list_cluster)

    def extend_cluster(self,cluster_number,idx,list_neighbours):
        list_seed = deepcopy(list_neighbours)
        while len(list_seed)> 0:
            #print("list_seed: {}".format(list_seed))
            idx_check = list_seed[0]
            list_seed.pop(0)
            if self.list_label[idx_check] == "noise":
                self.list_label[idx_check] = "border"
                self.update_cluster_assignment(cluster_number,idx_check)
            elif self.list_label[idx_check] != "unvisited":
                continue
            else:
                self.update_cluster_assignment(cluster_number,idx_check)
                new_neighbours = self.neighbours(self.X[:,[idx_check]])
                if len(new_neighbours)>= self.minpts:
                    self.list_label[idx_check] = "core"
                    list_seed = self.add_points(list_seed,new_neighbours)
                else:
                    self.list_label[idx_check] = "border"

    def fit(self,X):
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        cluster_number = -1
        for idx in range(self.nsample):
            # if point is processed already, then pass
            if self.list_label[idx] != "unvisited":
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
            self.update_cluster_assignment(cluster_number,idx)
            # build cluster
            self.extend_cluster(cluster_number,idx,list_neighbours)
        self.ncluster = cluster_number + 1
        self.time_fit = time.time() - time_start