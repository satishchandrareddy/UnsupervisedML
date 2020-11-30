# kmeans.py
# kmeans clustering

from copy import deepcopy
import create_data_cluster
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class kmeans:
    def __init__(self,ncluster, initialization='random'):
        self.ncluster = ncluster
        self.initialization = initialization
        self.clustersave = []
        self.list_objective = []

    def initialize_means(self):
        if self.initialization == 'kmeans++':
            # use k means ++ approach
            idx = np.random.randint(self.X.shape[1])
            self.mean = self.X[:,idx:idx+1]
            for count in range(1,self.ncluster):
                dist = self.compute_distance(self.X,self.mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                idx = np.argmax(np.min(dist,axis=0))                  
                self.mean = np.concatenate((self.mean,self.X[:,idx:idx+1]),axis=1)
        else:
            # initialize means at random
            self.mean = 0.3*np.random.randn(self.X.shape[0], self.ncluster)
        self.meansave = [deepcopy(self.mean)]

    def compute_distance(self,X,cluster):
        # compute distance between each sample in X and each mean in cluster
        ncluster = cluster.shape[1]
        dist = np.zeros((cluster.shape[1],X.shape[1]))
        for i in range(ncluster):
            dist[i,:] = np.sum(np.square(X-cluster[:,i:i+1]),axis=0)
        return dist

    def determine_cluster(self,dist):
        # determine cluster index for each point in data set and return objective function value
        self.cluster = np.argmin(dist,axis=0)
        self.clustersave.append(deepcopy(self.cluster))

    def compute_objective(self,dist):
        # compute sum of squares of distance to nearest cluster mean
        objective = np.sum(np.min(dist,axis=0))
        self.list_objective.append(objective)
        return objective

    def predict(self,X,cluster):
        # determine 
        dist = self.compute_distance(X,cluster)
        self.determine_cluster(dist)
        return self.cluster

    def check_diff(self):
        diff = np.sqrt(np.sum(np.square(self.meansave[-1] - self.meansave[-2])))
        return diff

    def update_mean(self):
        # loop over cluster
        for i in range(self.ncluster):
            # find points that are closest to current cluster mean
            idx = np.squeeze(np.where(np.absolute(self.cluster-i)<1e-7))
            if np.size(idx)==1:
                self.mean[:,i] = self.X[:,idx]
            elif np.size(idx)>1:
                self.mean[:,i] = np.sum(self.X[:,idx],axis=1)/np.size(idx)
        self.meansave.append(deepcopy(self.mean))

    def fit(self,X,nepoch,verbose=True):
        self.X = X
        self.initialize_means()
        # iterate to find cluster points
        diff = 10
        i = 0
        while (i< nepoch) and (diff>1e-7):
            i += 1
            # compute distances to all cluster means
            dist = self.compute_distance(self.X,self.mean)
            # determine cluster
            self.determine_cluster(dist)
            objective = self.compute_objective(dist)
            if verbose:
                print("Iteration: {}  Objective Function: {}".format(i,objective))
            # update_mean
            self.update_mean()
            diff = self.check_diff()
        return objective

    def get_mean(self):
        return self.mean

    def get_meansave(self):
        return self.meansave

    def plot_objective(self):
        fig = plt.subplots(1,1)
        list_iteration = list(range(0,len(self.list_objective)))
        plt.plot(list_iteration,self.list_objective,'b-')
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")

    def plot_cluster(self,X):
        # plot final clusters and means
        list_color = ["k", "r", "g", "m", "c"]
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_xlabel("Relative Salary")
        ax.set_ylabel("Relative Purchases")
        ax.set_title("Clusters")
        for cluster in range(self.ncluster):
            # plot cluster data
            symbol = list_color[cluster] + "o"
            idx = np.squeeze(np.where(np.absolute(self.clustersave[-1] - cluster)<1e-7))
            clusterdata, = plt.plot(X[0,idx],X[1,idx],symbol,markersize=4)
            # plot mean points 
            out, = plt.plot(self.meansave[-1][0,cluster],self.meansave[-1][1,cluster],color=list_color[cluster],marker ="s", markersize=8)

    def plot_results_animation(self,X,notebook=False):
        list_color = ["k", "r", "g", "m", "c"]
        fig,ax = plt.subplots(1,1)
        container = []
        original = True
        for count in range(len(self.meansave)):
            #iter_label_posy, iter_label_posx = ax.get_ylim()[1], ax.get_xlim()[1]
            #iteration_label = ax.text(0.85*iter_label_posx, 1.07*iter_label_posy,
            #        "", size=12, ha="center", animated=False)
            # plot data points ----- use separate frame
            ax.set_xlabel("Relative Salary")
            ax.set_ylabel("Relative Purchases")
            ax.set_title("Clusters")
            frame = []
            if original:
                # plot original data points in a single colour
                original = False
                originaldata, = plt.plot(X[0,:],X[1,:],"bo",markersize=4)
                frame.append(originaldata)
            else:
                # plot points for each cluster in separate colour
                for cluster in range(self.ncluster):
                    symbol = list_color[cluster] + "o"
                    idx = np.squeeze(np.where(np.absolute(self.clustersave[count-1] - cluster)<1e-7))
                    clusterdata, = plt.plot(X[0,idx],X[1,idx],symbol,markersize=4)
                    frame.append(clusterdata)
                    #iteration_label.set_text(f"Iteration: {count}")
                    #frame.append(iteration_label)
            container.append(frame)
            # ------
            # plot mean points ----- use separate frame
            for cluster in range(self.ncluster):
                out, = plt.plot(self.meansave[count][0,cluster],self.meansave[count][1,cluster],color=list_color[cluster],marker ="s", markersize=8)
                frame.append(out)
            container.append(frame)
        ani = animation.ArtistAnimation(fig,container, repeat = False, interval=350, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('cluster.mp4', writer='ffmpeg')
        if notebook:
            return ani
        plt.show()