# hierarchical.py

import clustering_base
from copy import deepcopy
import numpy as np
import time

class hierarchical(clustering_base.clustering_base):

    def initialize_algorithm(self):
        # initialize cluster information
        self.objectivesave = []
        self.clustersave = [(-1)*np.ones((self.nsample))]
        self.list_cluster = [[count] for count in range(self.nsample)]
        self.ncluster = self.nsample
        self.list_clustermean = [self.X[:,[count]] for count in range(self.nsample)]
        # for testing purposes
        self.list_clustersave = [deepcopy(self.list_cluster)]

    def fit(self,X):
        # determine clusters at all levels for dataset X
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        # loop over levels and combine closest clusters
        for level in range(self.nsample-1):
            self.combine_closest_clusters()
        self.time_fit = time.time() - time_start

    def combine_closest_clusters(self):
        # combine closest clusters at current level
        min_dist_squared = 1e100
        idx1 = None
        idx2 = None
        # loop over all pairs of non-empty clusters to determine closest clusters
        for count1 in range(self.nsample-1):
            for count2 in range(count1+1,self.nsample):
                if (len(self.list_cluster[count1])>0) and (len(self.list_cluster[count2])>0):
                    dist_squared = np.sum(np.square(self.list_clustermean[count1]-self.list_clustermean[count2]))
                    # if distance squared is smaller than than current smallest, save count1, count2 and dist squared
                    # idx1 is for cluster with larger number of points or first one in case of ties
                    if dist_squared<min_dist_squared:
                        if len(self.list_cluster[count1])>=len(self.list_cluster[count2]):
                            idx1 = count1
                            idx2 = count2
                        else:
                            idx1 = count2
                            idx2 = count1
                        min_dist_squared = dist_squared
        # add indices in idx2 to those in idx1, update saved cluster means, and empty entries in index idx2
        self.list_cluster[idx1].extend(self.list_cluster[idx2])
        self.list_cluster[idx2] = []
        self.list_clustermean[idx1] = np.mean(self.X[:,self.list_cluster[idx1]],axis=1,keepdims=True)
        self.list_clustermean[idx2] = []
        # update clustersave with new cluster assignment
        self.update_cluster_assignment(idx1)
        # for testing purposes
        self.list_clustersave.append(deepcopy(self.list_cluster))

    def update_cluster_assignment(self,idx1):
        # update clustersave assignment for latest updated cluster 
        array_clustersave = deepcopy(self.clustersave[-1])
        array_clustersave[self.list_cluster[idx1]] = self.list_cluster[idx1][0]
        self.clustersave.append(deepcopy(array_clustersave))