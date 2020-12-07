# hierarchical.py

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import normal
import numpy as np

class gaussianmm:
    def __init__(self,ncluster):
        self.clustersave = []
        self.ncluster = ncluster
        self.gammasave = []
        self.log_likelihood = []

    def initialize_parameters(self):
        # initialize means by picking data points at random
        mean = []
        array_idx = np.random.randint(low=0,high=self.X.shape[1],size=self.ncluster)
        for count in range(self.ncluster):
            mean.append(self.X[:,array_idx[count]:array_idx[count]+1])
        self.meansave = [mean]
        # initialize weights
        self.weightsave = [[1/self.ncluster for count in range(self.ncluster)]]
        # initialize covariance matrix
        Xmean = np.mean(self.X,axis=1,keepdims=True)
        Xmm = self.X-Xmean
        Sigma = np.dot(Xmm,Xmm.T)/self.nsample
        self.Sigmasave = [[Sigma for count in range(self.ncluster)]]
        print("mean: {}".format(self.meansave[-1]))
        print("Sigma: \n{}".format(self.Sigmasave))
        #print("weight: {}".format(self.weightsave[-1]))
        # initialize gamma
        self.update_gamma()
        # update log_likelihood
        self.update_log_likelihood() 
        # determine cluster
        self.determine_cluster()

    def update_gamma(self):
        self.weighted_normal = np.zeros((self.ncluster,self.nsample))
        for k in range(self.ncluster):
            self.weighted_normal[k,:] = self.weightsave[-1][k]*normal.normal_pdf(self.X,self.meansave[-1][k],self.Sigmasave[-1][k])
        self.gammasave.append(deepcopy(self.weighted_normal/np.sum(self.weighted_normal,axis=0,keepdims=True)))

    def update_log_likelihood(self):
        self.log_likelihood.append(np.sum(np.log(np.sum(self.weighted_normal,axis=0))))

    def update_parameters(self):
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
        print("list_weight: {}".format(list_weight))
        print("list mean: {}".format(list_mean))
        print("list Sigma: {}".format(list_Sigma))

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
        if verbose:
            print("Initial Log Likelihood: {}".format(self.log_likelihood[-1]))
        diff = 1e+10
        count = 1
        # loop over iteration
        while (count<=niteration and diff>1e-7):
            # update parameters
            self.update_parameters()
            # update gamma, log_likelihood, and cluster
            self.update_gamma()
            self.update_log_likelihood()
            self.determine_cluster()
            # print results
            if verbose:
                print("Log Likelihood Function: {}".format(self.log_likelihood[-1]))
            # compute difference:
            diff = self.compute_diff()
            count += 1

    def plot_objective(self):
        fig = plt.subplots(1,1)
        list_iteration = list(range(0,len(self.log_likelihood)))
        plt.plot(list_iteration,self.log_likelihood,'b-')
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")

    def plot_cluster(self,X):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_xlabel("Relative Salary")
        ax.set_ylabel("Relative Purchases")
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
        ax.set_xlabel("Relative Salary")
        ax.set_ylabel("Relative Purchases")
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