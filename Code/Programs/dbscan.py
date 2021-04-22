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
        # initialize cluster information
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
        # additional code added after video made in case no clusters found
        if self.ncluster == 0:
            self.ncluster = 1

    def neighbours(self,idx):
        # return list of indices of points within distance^2 <= epsilon^2 of point idx
        dist2 = np.sum(np.square(self.X[:,[idx]] - self.X),axis=0)
        return list(np.where(dist2<=self.epsilon2)[0])

    def extend_cluster(self,cluster_number,list_neighbour):
        # create a new cluster with label cluster_number and points (indices) in list_neighbour
        # list_s is list of  points (indices) in cluster
        # already_in_s tracks if point has been in list_s previously
        list_s = []
        already_in_s = [False for _ in range(self.nsample)]
        # add neighbours to list_s
        list_s,already_in_s = self.add_points_to_s(list_s,already_in_s,list_neighbour)
        while len(list_s)> 0:
            # check 0th point and then pop (remove)
            idx = list_s[0]
            list_s.pop(0)
            if self.list_label[idx] == "noise":
                self.list_label[idx] = "border"
                self.update_cluster_assignment(cluster_number,idx)
            elif self.list_label[idx] != "unvisited":
                continue
            else:
                self.update_cluster_assignment(cluster_number,idx)
                list_new_neighbour = self.neighbours(idx)
                if len(list_new_neighbour) < self.minpts:
                    self.list_label[idx] = "border"
                else:
                    self.list_label[idx] = "core"
                    list_s,already_in_s = self.add_points_to_s(list_s,already_in_s,list_new_neighbour)
     
    def add_points_to_s(self,list_s,already_in_s,list_neighbour):
        # add points from list_neighbour to list_s and return updated list_s and already_in_s
        for idx in list_neighbour:
            # add point only if not visited or noise and not previously in list_s
            if (self.list_label[idx]=="unvisited" or self.list_label[idx] == "noise") and (not already_in_s[idx]):
                # add point to list_s and update already_in_s
                list_s.append(idx)
                already_in_s[idx] = True
        return list_s, already_in_s

    def update_cluster_assignment(self,cluster_number,idx):
        # update clustersave with new cluster assignment: point idx in cluster = cluster_number
        if self.animation:
            current_clustersave = deepcopy(self.clustersave[-1])
            current_clustersave[idx] = cluster_number
            self.clustersave.append(current_clustersave)
        else:
            self.clustersave[-1][idx] = cluster_number