# dbscan.py

import clustering_base
from copy import deepcopy
import numpy as np
import time

class dbscan(clustering_base.clustering_base):
    def __init__(self,minpts,epsilon,animation=True):
        self.minpts = minpts
        self.epsilon = epsilon
        self.epsilon2 = epsilon**2
        self.animation = animation

    def initialize_algorithm(self):
        # initialize cluster information
        self.objectivesave = []
        self.list_label = ["unvisited" for _ in range(self.nsample)]
        self.clustersave = [(-1)*np.ones((self.nsample))]
        if not self.animation:
            self.clustersave.append(deepcopy(self.clustersave[0]))

    def fit(self,X):
        # perform dbscan algorithm
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
            list_neighbour = self.neighbours(idx)
            # determine if core point
            if len(list_neighbour) < self.minpts:
                self.list_label[idx] = "noise"
                continue
            # idx is a core point - start a new cluster (update cluster number and assignment)
            cluster_number += 1
            self.list_label[idx] = "core"
            self.update_cluster_assignment(cluster_number,idx)
            # extend cluster from core point
            self.extend_cluster(cluster_number,list_neighbour)
        self.ncluster = cluster_number + 1
        self.time_fit = time.time() - time_start

    def neighbours(self,idx):
        # return list of indices of points within distance epsilon of point idx
        dist = np.sum(np.square(self.X[:,[idx]] - self.X),axis=0)
        return np.where(dist<=self.epsilon2)[0]

    def extend_cluster(self,cluster_number,list_neighbour):
        # create a new cluster with label cluster_number and points (indices) in list_neighbour
        # list cluster is list of  points (indices) in cluster
        list_cluster = []
        # self.array_cluster tracks if point has been in list_cluster previously
        self.array_cluster = np.zeros((self.nsample))
        list_cluster = self.add_points(list_cluster,list_neighbour)
        while len(list_cluster)> 0:
            idx_check = list_cluster[0]
            list_cluster.pop(0)
            if self.list_label[idx_check] == "noise":
                self.list_label[idx_check] = "border"
                self.update_cluster_assignment(cluster_number,idx_check)
            elif self.list_label[idx_check] != "unvisited":
                continue
            else:
                self.update_cluster_assignment(cluster_number,idx_check)
                new_neighbours = self.neighbours(idx_check)
                if len(new_neighbours)>= self.minpts:
                    self.list_label[idx_check] = "core"
                    list_cluster = self.add_points(list_cluster,new_neighbours)
                else:
                    self.list_label[idx_check] = "border"
    
    def add_points(self,list_cluster,list_idx):
        # add points from list_idx to list_cluster and return updated list_cluster
        for idx in list_idx:
            # add point only if not visited or noise and not previously in list_cluster
            if (self.list_label[idx] == "unvisited" or self.list_label[idx] == "noise") and (self.array_cluster[idx]==0.0):
                # add point to cluster and update array cluster
                list_cluster.append(idx)
                self.array_cluster[idx] = 1.0
        return list_cluster

    def update_cluster_assignment(self,cluster_number,idx):
        # update clustersave with new cluster assignments
        if self.animation:
            list_cluster = deepcopy(self.clustersave[-1])
            list_cluster[idx] = cluster_number
            self.clustersave.append(list_cluster)
        else:
            self.clustersave[-1][idx] = cluster_number