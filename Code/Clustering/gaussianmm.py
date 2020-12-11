# gaussianmm.py

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import normal
import numpy as np
import pandas as pd

class gaussianmm:
    def __init__(self,ncluster,initialization="random"):
        self.ncluster = ncluster
        self.initialization = initialization
        self.clustersave = []
        self.gammasave = []
        self.loglikelihoodsave = []

    def initialize_parameters(self):
        # initialize means, covariances, and weights
        mean = []
        if self.initialization == "kmeans++":
            idx = np.random.randint(self.X.shape[1])
            mean.append(self.X[:,idx:idx+1])
            for count in range(1,self.ncluster):
                dist = self.compute_distance(self.X,mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                idx = np.argmax(np.min(dist,axis=0))                  
                mean.append(self.X[:,idx:idx+1])
        else:
            array_idx = np.random.randint(low=0,high=self.X.shape[1],size=self.ncluster)
            for count in range(self.ncluster):
                mean.append(self.X[:,array_idx[count]:array_idx[count]+1])
        # store in array
        self.meansave = [mean]
        # initialize weights
        self.weightsave = [[1/self.ncluster for _ in range(self.ncluster)]]
        # initialize covariance matrix
        Xmm = self.X - np.mean(self.X,axis=1,keepdims=True)
        Sigma = np.dot(Xmm,Xmm.T)/self.nsample
        self.Sigmasave = [[Sigma for _ in range(self.ncluster)]]

    def compute_distance(self,X,list_mean):
        # compute distance between each sample in X and each mean in cluster
        dist = np.zeros((len(list_mean),X.shape[1]))
        # loop over means in list_mean
        for count in range(len(list_mean)):
            dist[count,:] = np.sum(np.square(X-list_mean[count]),axis=0)
        return dist

    def expectation(self):
        # update gammas
        self.weighted_normal = np.zeros((self.ncluster,self.nsample))
        for k in range(self.ncluster):
            self.weighted_normal[k,:] = self.weightsave[-1][k]*normal.normal_pdf(self.X,self.meansave[-1][k],self.Sigmasave[-1][k])
        self.gammasave.append(deepcopy(self.weighted_normal/np.sum(self.weighted_normal,axis=0,keepdims=True)))
        # compute log likelihood and save
        self.loglikelihoodsave.append(np.sum(np.log(np.sum(self.weighted_normal,axis=0))))
        # determine clusters and save
        self.determine_cluster()

    def maximization(self):
        # compute number of points in each cluster
        self.N = np.sum(self.gammasave[-1],axis=1)
        # compute mean, Sigma, and weight for each cluster
        list_mean = []
        list_Sigma = []
        list_weight = []
        for k in range(self.ncluster):
            mean = np.sum(self.X*self.gammasave[-1][k,:],axis=1,keepdims=True)/self.N[k]
            list_mean.append(mean)
            list_weight.append(self.N[k]/self.nsample)
            Xmm = self.X - mean
            list_Sigma.append(np.dot(self.gammasave[-1][k,:]*Xmm,Xmm.T)/self.N[k])
        self.meansave.append(list_mean)
        self.weightsave.append(list_weight)
        self.Sigmasave.append(list_Sigma)

    def compute_diff(self):
        # determine sum of distances between current and previous means
        diff = 0
        for k in range(self.ncluster):
            diff += np.sqrt(np.sum(np.square(self.meansave[-1][k]-self.meansave[-2][k])))
        return diff

    def determine_cluster(self):
        # determine cluster index for each point in data set
        self.clustersave.append(np.argmax(self.gammasave[-1],axis=0))

    def get_meansave(self):
        return self.meansave

    def fit(self,X,niteration,verbose=True):
        self.X = X
        self.nsample = X.shape[1]
        # initialize
        self.initialize_parameters()
        diff = 1e+10
        count = 1
        # loop over iteration
        while (count<=niteration and diff>1e-7):
            # expectation step
            self.expectation()
            # maximization step
            self.maximization()
            # print results
            if verbose:
                print("Log Likelihood Function: {}".format(self.loglikelihoodsave[-1]))
            # compute difference:
            diff = self.compute_diff()
            count += 1
        return self.loglikelihoodsave

    def plot_cluster(self,X):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        ax.set_title("Clusters")
        color_multiplier = 1/(self.ncluster)
        for cluster in range(self.ncluster):
            # plot cluster data
            idx = np.squeeze(np.where(np.absolute(self.clustersave[-1] - cluster)<1e-7))
            color = color_multiplier*(cluster+1)
            clusterdata = plt.scatter(X[0,idx],X[1,idx],color=cm.jet(color),marker="o",s=20)
            # plot mean points 
            mean = plt.scatter(self.meansave[-1][cluster][0,0],self.meansave[-1][cluster][1,0],color=cm.jet(color),marker ="s", s=50)

    def plot_results_animation(self,X,notebook=False):
        fig,ax = plt.subplots(1,1)
        container = []
        original = True
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        ax.set_title("Clusters")
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
                    clusterdata = plt.scatter(X[0,idx],X[1,idx],color=cm.jet(color),marker="o",s=20)
                    frame.append(clusterdata)
            container.append(frame)
            # plot mean points ----- use separate frame
            for cluster in range(self.ncluster):
                color = color_multiplier*(cluster+1)
                mean = plt.scatter(self.meansave[count][cluster][0,0],self.meansave[count][cluster][1,0],color=cm.jet(color),marker ="s",s=50)
                frame.append(mean)
            container.append(frame)
        ani = animation.ArtistAnimation(fig,container, repeat = False, interval=350, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('cluster.mp4', writer='ffmpeg')
        if notebook:
            return ani
        plt.show()

    @staticmethod
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[0, 1], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
        return ax

    def plot_ellipse_animation(self, X, num_frames, notebook=False):
        def update(frame_num, gmm, X, scat, ax=None):
            weights = np.array(gmm.weightsave[frame_num])
            means = np.squeeze(np.array(gmm.meansave[frame_num]))
            covars = np.array(gmm.Sigmasave[frame_num])
            ax = ax or plt.gca()
            ax.clear()
            labels = gmm.clustersave[frame_num]
            ax.axis('equal')
            w_factor = 0.2 / weights.max()
            for pos, covar, w in zip(means,covars,weights):
              ax.scatter(X[0,:], X[1,:], c=labels)
              ax = gmm.draw_ellipse(pos, covar, alpha=w * w_factor)
                
            return scat,

        fig = plt.figure()
        scat = plt.scatter(X[0,:], X[1,:])
        ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, 
                                fargs=(self, X, scat), interval=500, blit=True)
        if notebook:
            return ani
        plt.show()


    def plot_cluster_distribution(self, labels, figsize=(12,4)):
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
