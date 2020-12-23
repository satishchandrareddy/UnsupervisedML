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
            diff = max(diff,np.sqrt(np.sum(np.square(self.meansave[-1][k]-self.meansave[-2][k]))))
        return diff

    def determine_cluster(self):
        # determine cluster index for each point in data set
        self.clustersave.append(np.argmax(self.gammasave[-1],axis=0))

    def get_meansave(self):
        return self.meansave

    def fit(self,X,niteration,tolerance=1e-5,verbose=True):
        self.X = X
        self.nsample = X.shape[1]
        # initialize
        self.initialize_parameters()
        diff = 1e+10
        count = 1
        # loop over iteration
        while (count<=niteration and diff>tolerance):
            # expectation step
            self.expectation()
            # maximization step
            self.maximization()
            # print results
            if verbose:
                print("Iteration: {} - Log Likelihood Function: {}".format(count,self.loglikelihoodsave[-1]))
            # compute difference:
            diff = self.compute_diff()
            count += 1
        return self.loglikelihoodsave

    def plot_cluster(self):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        ax.set_title("Clusters and Gaussians")
        array_color_data = (1+self.clustersave[-1])/self.ncluster
        scatter_data = plt.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)
        # plot Gaussian footprints
        for cluster in range(self.ncluster):
            mean, width, height, angle = normal.create_ellipse_patch_details(self.meansave[-1][cluster],self.Sigmasave[-1][cluster],self.weightsave[-1][cluster])
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color=cm.jet((cluster+1)/self.ncluster), alpha=0.5)
            ax.add_patch(ell)

    def plot_results_animation(self,X,notebook=False):
        fig,ax = plt.subplots(1,1)
        ax.set_title("Evolution of Clusters")
        # create dummy ellipse containers
        list_object = []
        for cluster in range(self.ncluster):
            ell = Ellipse(xy=np.array([0,0]), width=1, height=1, angle=0, color=cm.jet((cluster+1)/self.ncluster), alpha=0.5, visible=False)
            list_object.append(ell)
            ax.add_patch(ell)
        # scatter plot of data
        scat = ax.scatter(X[0,:], X[1,:], color = cm.jet(0), marker="o", s=15)
        list_object.append(scat)

        def init():
            global list_object
            list_object[-1].scatter(X[0,:], X[1,:], color = cm.jet(0), marker="o", s=15)
            return list_object

        def update(i,list_object,clustersave,meansave,Covsave,weightsave):
            # update ellipse details for each cluster
            nellipse = len(list_object)-1
            for cluster in range(nellipse):
                mean, width, height, angle = normal.create_ellipse_patch_details(meansave[i][cluster],
                    Covsave[i][cluster],weightsave[i][cluster])
                list_object[cluster].set_center(mean)
                list_object[cluster].width = width
                list_object[cluster].height = height
                list_object[cluster].angle = angle
                list_object[cluster].set_visible(True)
            if i == 0:
                list_object[-1].set_color(cm.jet(0))
            else:
                list_object[-1].set_color(cm.jet((np.squeeze(clustersave[i])+1)/(nellipse)))
            return list_object

        ani = animation.FuncAnimation(fig=fig, func=update, frames = len(self.clustersave),
            fargs=[list_object,self.clustersave,self.meansave,self.Sigmasave,self.weightsave],
            repeat_delay=1000, repeat=True, interval=1000, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        ani.save('cluster.mp4', writer='ffmpeg')
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