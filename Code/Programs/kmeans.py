# kmeans.py

import clustering_base
from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

class kmeans(clustering_base.clustering_base):
    def __init__(self,ncluster, initialization='random'):
        self.ncluster = ncluster
        self.initialization = initialization

    def initialize_algorithm(self):
        # initialize cluster information
        self.objectivesave = []
        self.clustersave = [(-1)*np.ones((self.nsample))]
        if self.initialization == 'kmeans++':
            # use k means++ approach
            # compute first mean
            idx = np.random.randint(self.X.shape[1])
            mean = [self.X[:,[idx]]]
            # compute means 2,..,K
            for count in range(1,self.ncluster):
                dist2 = self.compute_distance2(mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                idx = np.argmax(np.min(dist2,axis=0))                 
                mean.append(self.X[:,[idx]])
        else:
            # pick initial means at random
            # randomly pick ncluster indices from 0,1,2,...,nsample-1 and create list of initial means
            array_idx = np.random.choice(self.X.shape[1],self.ncluster)
            mean = [self.X[:,[array_idx[count]]] for count in range(self.ncluster)]
        # meansave is list of list of means
        self.meansave = [mean]

    def fit(self,X,max_iter,tolerance=1e-5,verbose=True):
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        self.initialize_algorithm()
        # iterate to find cluster means and assignments
        diff = 10
        i = 0
        while (i< max_iter) and (diff>tolerance):
            # compute distance squared to all cluster means
            dist2 = self.compute_distance2(self.meansave[-1])
            # update cluster assignments
            self.update_cluster_assignment(dist2)
            # compute latest objective function value and print
            self.update_objective(dist2)
            if verbose:
                print("Iteration: {}  Objective Function: {}".format(i,self.objectivesave[-1]))
            # update means
            self.update_mean()
            # determine max difference between current and previous means
            diff = self.compute_diff_mean()
            i += 1
        self.time_fit = time.time() - time_start

    def compute_distance2(self,list_mean):
        # dist2[i,j] = distance squared between mean i and point j 
        dist2 = np.zeros((len(list_mean),self.X.shape[1]))
        # compute dist2 one row at at time (distance squared to cluster mean i for all points) 
        for count in range(len(list_mean)):
            dist2[count,:] = np.sum(np.square(self.X-list_mean[count]),axis=0)
        return dist2

    def update_cluster_assignment(self,dist2):
        # update cluster_assisgnment based on distance squared matrix
        self.clustersave.append(np.argmin(dist2,axis=0))

    def update_objective(self,dist2):
        # update objective function = sum of squares of distance to nearest cluster mean
        self.objectivesave.append(np.sum(np.min(dist2,axis=0)))

    def update_mean(self):
        # update means based on latest cluster assignments
        list_mean = deepcopy(self.meansave[-1])
        for count in range(self.ncluster):
            # for each cluster find indices of points assigned to the cluster
            idx = np.where(np.absolute(self.clustersave[-1]-count)<1e-7)[0]
            # update mean if there are points assigned to cluster
            if np.size(idx)>0:
                list_mean[count] = np.mean(self.X[:,idx],axis=1,keepdims=True)
        self.meansave.append(deepcopy(list_mean))

    def compute_diff_mean(self):
        # determine maximum distance between current and previous means
        diff = 0
        for count in range(self.ncluster):
            diff = max(diff,np.sqrt(np.sum(np.square(self.meansave[-1][count]-self.meansave[-2][count]))))
        return diff

    def plot_cluster(self,nlevel,title="",xlabel="",ylabel=""):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # plot each mean and associated data points in same color
        array_mean = np.concatenate(tuple(self.meansave[nlevel]),axis=1)
        array_color_data = (1+self.clustersave[nlevel])/self.ncluster
        array_color_mean = (1+np.arange(self.ncluster))/self.ncluster
        scatter_data = plt.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)
        scatter_mean = plt.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color_mean), marker="s", s=50)

    def plot_cluster_animation(self,nlevel=-1,interval=50,title="",xlabel="",ylabel=""):
        # animation of evolutions of means and cluster assignments
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel
        # create initial scatter objects for means and data
        array_mean = np.concatenate(tuple(self.meansave[0]),axis=1)
        array_color_mean = (1+np.arange(self.ncluster))/self.ncluster
        scat_data = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)
        scat_mean = ax.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color_mean),marker="s",s=50)
        # create update function - update color for data points and locations of means
        def update(i,scat_data,scat_mean,clustersave,meansave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat_data.set_color(cm.jet(array_color_data))
            array_mean = np.concatenate(tuple(self.meansave[i]),axis=1)
            scat_mean.set_offsets(array_mean.T)
            return scat_data,scat_mean
        # create animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[scat_data,scat_mean,self.clustersave,self.meansave,self.ncluster],
            repeat_delay=1000, interval=interval, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get details
        #ani.save('KMeans_Animation.mp4', writer='ffmpeg')