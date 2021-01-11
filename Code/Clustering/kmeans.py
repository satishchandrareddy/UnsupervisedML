# kmeans.py
# kmeans clustering

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import time

class kmeans:
    def __init__(self,ncluster, initialization='random'):
        self.ncluster = ncluster
        self.initialization = initialization

    def initialize_algorithm(self):
        self.objectivesave = []
        self.clustersave = [(-1)*np.ones((self.nsample))]
        if self.initialization == 'kmeans++':
            # use k means ++ approach
            idx = np.random.randint(self.X.shape[1])
            mean = [self.X[:,idx:idx+1]]
            for count in range(1,self.ncluster):
                dist = self.compute_distance(mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
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
        return self.objectivesave

    def plot_cluster(self,title="",xlabel="",ylabel=""):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        array_mean = np.concatenate(tuple(self.meansave[-1]),axis=1)
        array_color_data = (1+self.clustersave[-1])/self.ncluster
        array_color_mean = np.arange(1,self.ncluster+1)/self.ncluster
        scatter_data = plt.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)
        scatter_mean = plt.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color_mean), marker = "s", s=50)

    def plot_results_animation(self,title="",xlabel="",ylabel=""):
        fig,ax = plt.subplots(1,1)
        container = []
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # loop over iterations
        for count in range(len(self.meansave)):
            # plot data points ----- use separate frame
            frame = []
            array_color = (self.clustersave[count]+1)/self.ncluster
            clusterdata = plt.scatter(self.X[0,:],self.X[1,:],color=cm.jet(array_color),marker="o",s=15)
            frame.append(clusterdata)
            container.append(frame)
            # plot mean points ----- use separate frame
            array_color = np.arange(1,self.ncluster+1)/self.ncluster
            array_mean = np.concatenate(tuple(self.meansave[count]),axis=1)
            mean = plt.scatter(array_mean[0,:],array_mean[1,:],color=cm.jet(array_color),marker ="s",s=50)
            frame.append(mean)
            container.append(frame)
        ani = animation.ArtistAnimation(fig,container, repeat = False, interval=1000, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('kmeans.mp4', writer='ffmpeg')

    def plot_cluster_distribution(self, labels, figsize=(25,4)):
          print(f"Number of Clusters: {self.ncluster}")
          cluster_labels = self.clustersave[-1]
          df = pd.DataFrame({'class': labels,
                            'cluster label': cluster_labels,
                            'cluster': np.ones(len(labels))})
          counts = df.groupby(['cluster label', 'class']).sum()
          fig = counts.unstack(level=0).plot(kind='bar', subplots=True,
                                            sharey=True, sharex=False,
                                            layout=(1,self.ncluster), 
                                            figsize=figsize, legend=False)
          plt.show()