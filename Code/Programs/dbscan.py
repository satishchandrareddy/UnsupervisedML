# dbscan.py

import clustering_base
from copy import deepcopy
import numpy as np
import time

class dbscan(clustering_base.clustering_base):
    def __init__(self,minpts,epsilon,animation=True):
        self.minpts = minpts
        self.epsilon2 = epsilon**2
        self.animation = animation

    def initialize_algorithm(self):
        self.objectivesave = []
        # initialize cluste information
        self.list_label = ["unvisited" for _ in range(self.nsample)]
        self.clustersave = [(-1)*np.ones((self.nsample))]
        if not self.animation:
            # create self.clustersave[1] for final cluster assignment in case of no animation
            self.clustersave.append(deepcopy(self.clustersave[0]))

    def fit(self,X):
        # perform dbscan algorithm
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        cluster_number = -1
        for idx in range(self.nsample):
            # if point is processed already, then continue
            if self.list_label[idx] != "unvisited":
                continue
            # determine neighbours
            list_neighbour = self.neighbours(idx)
            # determine if noise => no need to update cluster assignment as assignment remains -1
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
        # return list of indices of points within distance^2 = epsilon^2 of point idx
        dist2 = np.sum(np.square(self.X[:,[idx]] - self.X),axis=0)
        return list(np.where(dist2<=self.epsilon2)[0])

    def extend_cluster(self,cluster_number,list_neighbour):
        # create a new cluster with label cluster_number and points (indices) in list_neighbour
        # list cluster is list of  points (indices) in cluster
        list_cluster = []
        # in_cluster tracks if point has been in list_cluster previously
        in_cluster = [False for _ in range(self.nsample)]
        list_cluster,in_cluster = self.add_points(list_cluster,in_cluster,list_neighbour)
        while len(list_cluster)> 0:
            # check 0th point and then pop (remove)
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
                    list_cluster,in_cluster = self.add_points(list_cluster,in_cluster,new_neighbours)
                else:
                    self.list_label[idx_check] = "border"
    
    def add_points(self,list_cluster,in_cluster,list_idx):
        # add points from list_idx to list_cluster and return updated list_cluster
        for idx in list_idx:
            # add point only if not visited or noise and not previously in list_cluster
            if (self.list_label[idx] == "unvisited" or self.list_label[idx] == "noise") and (not in_cluster[idx]):
                # add point to cluster and update in_cluster
                list_cluster.append(idx)
                in_cluster[idx] = True
        return list_cluster, in_cluster

    def update_cluster_assignment(self,cluster_number,idx):
        # update clustersave with new cluster assignment
        if self.animation:
            current_clustersave = deepcopy(self.clustersave[-1])
            current_clustersave[idx] = cluster_number
            self.clustersave.append(current_clustersave)
        else:
            self.clustersave[-1][idx] = cluster_number