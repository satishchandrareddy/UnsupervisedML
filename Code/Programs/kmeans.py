# kmeans.py

import clustering_base
from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import time

class kmeans(clustering_base.clustering_base):
    def __init__(self,ncluster, initialization='random'):
        self.ncluster = ncluster
        self.initialization = initialization

    def initialize_algorithm(self):
        self.objectivesave = []
        self.clustersave = [(-1)*np.ones((self.nsample))]
        if self.initialization == 'kmeans++':
            # use k means ++ approach
            idx = np.random.randint(self.X.shape[1])
            mean = [self.X[:,[idx]]]
            for count in range(1,self.ncluster):
                dist = self.compute_distance(mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                junk = np.min(dist,axis=0)
                junk_argsort = np.argsort(-junk)[0:5]
                idx = np.argmax(np.min(dist,axis=0))                 
                mean.append(self.X[:,[idx]])
        else:
            # initialize means at random
            mean = []
            array_idx = np.random.choice(self.X.shape[1],self.ncluster)
            for count in range(self.ncluster):
                mean.append(self.X[:,[array_idx[count]]])
        self.meansave = [mean]

    def compute_distance(self,list_mean):
        # compute distance squared between each sample in X and each mean in cluster
        dist = np.zeros((len(list_mean),self.X.shape[1]))
        # loop over means in list_mean
        for count in range(len(list_mean)):
            dist[count,:] = np.sum(np.square(self.X-list_mean[count]),axis=0)
        return dist

    def update_cluster_assignment(self,dist):
        # update cluster_assignment for this iteration
        self.clustersave.append(np.argmin(dist,axis=0))

    def compute_objective(self,dist):
        # compute sum of squares of distance to nearest cluster mean
        self.objectivesave.append(np.sum(np.min(dist,axis=0)))

    def compute_diff_mean(self):
        # determine sum of distances between current and previous means
        diff = 0
        for count in range(self.ncluster):
            diff = max(diff,np.sqrt(np.sum(np.square(self.meansave[-1][count]-self.meansave[-2][count]))))
        return diff

    def update_mean(self):
        # loop over cluster
        list_mean = deepcopy(self.meansave[-1])
        for count in range(self.ncluster):
            # find points that are closest to current cluster mean
            idx = np.where(np.absolute(self.clustersave[-1]-count)<1e-7)[0]
            # update mean if there are points assigned to cluster
            if np.size(idx)>1:
                list_mean[count] = np.mean(self.X[:,idx],axis=1,keepdims=True)
        # save current means
        self.meansave.append(deepcopy(list_mean))

    def fit(self,X,niteration,tolerance=1e-5,verbose=True):
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        # iterate to find cluster points
        diff = 10
        i = 0
        while (i< niteration) and (diff>tolerance):
            # compute distances to all cluster means
            dist = self.compute_distance(self.meansave[-1])
            #print("dist: {}".format(dist))
            # determine cluster
            self.update_cluster_assignment(dist)
            self.compute_objective(dist)
            if verbose:
                print("Iteration: {}  Objective Function: {}".format(i,self.objectivesave[-1]))
            # update_mean
            self.update_mean()
            diff = self.compute_diff_mean()
            i += 1
        time_end = time.time()
        print("K Means fit time: {}".format(time_end - time_start))
        return time_end-time_start

    def plot_cluster(self,nlevel,title="",xlabel="",ylabel=""):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        array_mean = np.concatenate(tuple(self.meansave[nlevel]),axis=1)
        array_color_data = (1+self.clustersave[nlevel])/self.ncluster
        array_color_mean = np.arange(1,self.ncluster+1)/self.ncluster
        scatter_data = plt.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)
        scatter_mean = plt.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color_mean), marker="s", s=50)

    def plot_cluster_animation(self,nlevel=-1,interval=50,title="",xlabel="",ylabel=""):
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel

        array_mean = np.concatenate(tuple(self.meansave[0]),axis=1)
        array_color_mean = np.arange(1,self.ncluster+1)/self.ncluster
        scat_data = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)
        scat_mean = ax.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color_mean),marker="s",s=50)

        def update(i,scat_data,scat_mean,X,clustersave,meansave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat_data.set_color(cm.jet(array_color_data))
            array_mean = np.concatenate(tuple(self.meansave[i]),axis=1)
            array_color_mean = np.arange(1,self.ncluster+1)/self.ncluster
            scat_mean.set_color(cm.jet(array_color_mean))
            scat_mean.set_offsets(array_mean.T)
            return scat_data,scat_mean

        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[scat_data,scat_mean,self.X,self.clustersave,self.meansave,self.ncluster], repeat_delay=1000, repeat=True, interval=interval, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        ani.save('KMeans_Animation.mp4', writer='ffmpeg')
        plt.show()