# hierarchical.py
# hierarchical clustering

from copy import deepcopy
import clustering_base
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

class hierarchical(clustering_base.clustering_base):

    def initialize_algorithm(self):
        # initialize cluster information
        self.clustersave = [(-1)*np.ones((self.nsample))]
        self.list_cluster = [[count] for count in range(self.nsample)]
        self.ncluster = self.nsample

    def dist_between_clusters(self,idx1,idx2):
        # idx1, idx2 are lists of indices of data set
        # this function returns distance squared between idx1 and idx2 points
        return np.sum(np.square(np.mean(self.X[:,idx1],axis=1)-np.mean(self.X[:,idx2],axis=1)))

    def update_cluster_assignment(self,idx1,idx2):
        # idx1, idx2 are lists of indices of data set
        # update clustersave
        array_clustersave = deepcopy(self.clustersave[-1])
        array_clustersave[self.list_cluster[idx1]] = self.list_cluster[idx1][0]
        array_clustersave[self.list_cluster[idx2]] = self.list_cluster[idx1][0]
        self.clustersave.append(deepcopy(array_clustersave))

    def combine_closest_clusters(self):
        # combine closest clusters at current level
        cluster_dist = 1e100
        idx1 = None
        idx2 = None
        # loop over all pairs of clusters to determine closest clusters
        for count1 in range(self.nsample-1):
            for count2 in range(count1+1,self.nsample):
                if (len(self.list_cluster[count1])>0) and (len(self.list_cluster[count2])>0):
                    dist = self.dist_between_clusters(self.list_cluster[count1],self.list_cluster[count2])
                    # if distance is smaller than than current smallest, save 
                    if dist<cluster_dist:
                        if len(self.list_cluster[count1])>len(self.list_cluster[count2]):
                            idx1 = count1
                            idx2 = count2
                        else:
                            idx1 = count2
                            idx2 = count1
                        cluster_dist = dist
        # add indices in idx2 to those in idx1 and empty indices in idx2
        self.list_cluster[idx1].extend(self.list_cluster[idx2])
        self.list_cluster[idx2] = []
        # update clustersave with new cluster assignment
        self.update_cluster_assignment(idx1,idx2)

    def fit(self,X):
        # X is dataset
        # main function for determining cluster assignments
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        # go through process of combining nearest clusters at each level
        for level in range(self.nsample-1):
            self.combine_closest_clusters()
        time_end = time.time()
        print("Time Hierarchical: {}".format(time_end - time_start))