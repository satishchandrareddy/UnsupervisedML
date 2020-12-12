# kmeans.py
# kmeans clustering

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

class kmeans:
    def __init__(self,ncluster, initialization='random'):
        self.ncluster = ncluster
        self.initialization = initialization
        self.clustersave = []
        self.objectivesave = []

    def initialize_means(self):
        if self.initialization == 'kmeans++':
            # use k means ++ approach
            idx = np.random.randint(self.X.shape[1])
            self.mean = [self.X[:,idx:idx+1]]
            for count in range(1,self.ncluster):
                dist = self.compute_distance(self.X,self.mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                idx = np.argmax(np.min(dist,axis=0))                  
                self.mean.append(self.X[:,idx:idx+1])
        else:
            # initialize means at random
            self.mean = []
            array_idx = np.random.choice(self.X.shape[1],self.ncluster)
            for count in range(self.ncluster):
                self.mean.append(self.X[:,array_idx[count]:array_idx[count]+1])
        self.meansave = [deepcopy(self.mean)]

    def compute_distance(self,X,list_mean):
        # compute distance between each sample in X and each mean in cluster
        dist = np.zeros((len(list_mean),X.shape[1]))
        # loop over means in list_mean
        for count in range(len(list_mean)):
            dist[count,:] = np.sum(np.square(X-list_mean[count]),axis=0)
        return dist

    def determine_cluster(self,dist):
        # determine cluster index for each point in data set and return objective function value
        self.cluster = np.argmin(dist,axis=0)
        self.clustersave.append(deepcopy(self.cluster))

    def compute_objective(self,dist):
        # compute sum of squares of distance to nearest cluster mean
        objective = np.sum(np.min(dist,axis=0))
        self.objectivesave.append(objective)

    def check_diff(self):
        # determine sum of distances between current and previous means
        diff = 0
        for count in range(self.ncluster):
            diff = np.sqrt(np.sum(np.square(self.meansave[-1][count]-self.meansave[-2][count])))
        return diff

    def update_mean(self):
        # loop over cluster
        for count in range(self.ncluster):
            # find points that are closest to current cluster mean
            idx = np.squeeze(np.where(np.absolute(self.cluster-count)<1e-7))
            # compute mean of points and save in list
            if np.size(idx)==1:
                self.mean[count] = self.X[:,idx:idx+1]
            elif np.size(idx)>1:
                self.mean[count] = np.mean(self.X[:,idx],axis=1,keepdims=True)
        # save current means
        self.meansave.append(deepcopy(self.mean))

    def fit(self,X,niteration,verbose=True):
        self.X = X
        self.initialize_means()
        # iterate to find cluster points
        diff = 10
        i = 0
        while (i< niteration) and (diff>1e-7):
            # compute distances to all cluster means
            dist = self.compute_distance(self.X,self.mean)
            # determine cluster
            self.determine_cluster(dist)
            self.compute_objective(dist)
            if verbose:
                print("Iteration: {}  Objective Function: {}".format(i,self.objectivesave[-1]))
            # update_mean
            self.update_mean()
            diff = self.check_diff()
            i += 1
        return self.objectivesave

    def get_mean(self):
        return self.mean

    def get_meansave(self):
        return self.meansave

    def plot_cluster(self,X,**kwargs):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        #ax.set_xlabel("Relative Salary")
        #ax.set_ylabel("Relative Purchases")
        ax.set_title("Clusters and Means")
        color_multiplier = 1/(self.ncluster)
        for cluster in range(self.ncluster):
            # plot cluster data
            idx = np.squeeze(np.where(np.absolute(self.clustersave[-1] - cluster)<1e-7))
            color = color_multiplier*(cluster+1)
            clusterdata = plt.scatter(X[0,idx],X[1,idx],color=cm.jet(color),marker="o",s=15)
            # plot mean points 
            mean = plt.scatter(self.meansave[-1][cluster][0,0],self.meansave[-1][cluster][1,0],color=cm.jet(color),marker ="s", s=50)

    def plot_results_animation(self,X,notebook=False):
        fig,ax = plt.subplots(1,1)
        container = []
        original = True
        #ax.set_xlabel("Relative Salary")
        #ax.set_ylabel("Relative Purchases")
        ax.set_title("Evolution of Clusters and Means")
        color_multiplier = 1/self.ncluster
        # loop over iterations
        for count in range(len(self.meansave)):
            # plot data points ----- use separate frame
            frame = []
            if original: # plot original data points in a single colour
                original = False
                originaldata = plt.scatter(X[0,:],X[1,:],color=cm.jet(0),marker="o",s=20)
                frame.append(originaldata)
            else: # plot points for each cluster in separate colour
                for cluster in range(self.ncluster):
                    color = color_multiplier*(cluster+1)
                    idx = np.squeeze(np.where(np.absolute(self.clustersave[count-1] - cluster)<1e-7))
                    clusterdata = plt.scatter(X[0,idx],X[1,idx],color=cm.jet(color),marker="o",s=15)
                    frame.append(clusterdata)
            container.append(frame)
            # plot mean points ----- use separate frame
            for cluster in range(self.ncluster):
                color = color_multiplier*(cluster+1)
                mean = plt.scatter(self.meansave[count][cluster][0,0],self.meansave[count][cluster][1,0],color=cm.jet(color),marker ="s",s=50)
                frame.append(mean)
            container.append(frame)
        ani = animation.ArtistAnimation(fig,container, repeat = False, interval=1000, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        ani.save('kmeans.mp4', writer='ffmpeg')
        if notebook:
            return ani
        plt.show()

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