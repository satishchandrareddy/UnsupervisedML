# gaussianmm.py

import clustering_base
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import normal
import numpy as np
import time

class gaussianmm(clustering_base.clustering_base):
    def __init__(self,ncluster,initialization="random"):
        self.ncluster = ncluster
        self.initialization = initialization

    def initialize_algorithm(self):
        # determine initial means, covariances, and weights for gaussians in mixture
        self.clustersave = [(-1)*np.ones((self.nsample))]
        self.objectivesave = []
        # initialize means using "kmeans++" or "random" approaches
        mean = []
        if self.initialization == "kmeans++":
            idx = np.random.randint(self.X.shape[1])
            mean.append(self.X[:,idx:idx+1])
            for count in range(1,self.ncluster):
                dist2 = self.compute_distance2(mean)
                # pick data point whose distance squared from nearest cluster mean is greatest
                idx = np.argmax(np.min(dist2,axis=0))                  
                mean.append(self.X[:,[idx]])
        else:
            # pick initial means at random
            # randomly pick ncluster indices from 0,1,2,...,nsample-1 and create list of initial means
            array_idx = np.random.choice(self.X.shape[1],self.ncluster,replace=False)
            mean = [self.X[:,[array_idx[count]]] for count in range(self.ncluster)]
        # meansave is list of list of means
        self.meansave = [mean]
        # initialize weights = 1/# of clusters for all gaussians
        self.weightsave = [[1/self.ncluster for _ in range(self.ncluster)]]
        # initialize covariance matrices as same for all gaussians
        Xmm = self.X - np.mean(self.X,axis=1,keepdims=True)
        Cov = np.matmul(Xmm,Xmm.T)/self.nsample
        self.Covsave = [[Cov for _ in range(self.ncluster)]]

    def fit(self,X,max_iter,tolerance=1e-5,verbose=True):
        time_start = time.time()
        self.X = X
        self.nsample = X.shape[1]
        # initialize
        self.initialize_algorithm()
        diff = 1e+10
        count = 0
        # loop 
        while (count<max_iter and diff>tolerance):
            # expectation step
            self.expectation()
            # update cluster assignment
            self.update_cluster_assignment()
            # maximization step
            self.maximization()
            # print results
            if verbose:
                print("Iteration: {} - Log Likelihood Function: {}".format(count,self.objectivesave[-1]))
            # compute difference:
            diff = self.compute_diff()
            count += 1
        self.time_fit = time.time() - time_start

    def expectation(self):
        # expectation step - update conditional probabilities
        weighted_normal = np.zeros((self.ncluster,self.nsample))
        for k in range(self.ncluster):
            weighted_normal[k,:] = self.weightsave[-1][k]*normal.normal_pdf_vectorized(self.X,self.meansave[-1][k],self.Covsave[-1][k])
        self.gamma = weighted_normal/np.sum(weighted_normal,axis=0,keepdims=True)
        # compute log likelihood and append to objectivesave
        self.objectivesave.append(np.sum(np.log(np.sum(weighted_normal,axis=0))))

    def maximization(self):
        # compute number of points in each cluster
        self.M = np.sum(self.gamma,axis=1)
        # compute mean, Cov, and weight for each cluster
        list_mean = []
        list_Cov = []
        list_weight = []
        for k in range(self.ncluster):
            mean = np.sum(self.X*self.gamma[k,:],axis=1,keepdims=True)/self.M[k]
            list_mean.append(mean)
            list_weight.append(self.M[k]/self.nsample)
            Xmm = self.X - mean
            list_Cov.append(np.matmul(self.gamma[k,:]*Xmm,Xmm.T)/self.M[k])
        self.meansave.append(list_mean)
        self.weightsave.append(list_weight)
        self.Covsave.append(list_Cov)

    def compute_distance2(self,list_mean):
        # dist2[i,j] = distance squared between mean i and point j 
        dist2 = np.zeros((len(list_mean),self.X.shape[1]))
        # loop over means in list_mean
        for count in range(len(list_mean)):
            dist2[count,:] = np.sum(np.square(self.X-list_mean[count]),axis=0)
        return dist2

    def update_cluster_assignment(self):
        # determine cluster assignment = index of maximum value of gamma for each data point
        self.clustersave.append(np.argmax(self.gamma,axis=0))

    def compute_diff(self):
        # determine maximum difference between current and previous means
        diff = 0
        for k in range(self.ncluster):
            diff = max(diff,np.sqrt(np.sum(np.square(self.meansave[-1][k]-self.meansave[-2][k]))))
        return diff

    def plot_cluster(self,nlevel,title="",xlabel="",ylabel=""):
        # plot final clusters and means
        fig,ax = plt.subplots(1,1)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        array_color_data = (1+self.clustersave[nlevel])/self.ncluster
        scatter_data = ax.scatter(self.X[0,:],self.X[1,:], color=cm.jet(array_color_data), marker="o", s=15)
        # new code: elliptical patch of weighted normal pdf for clusters
        for cluster in range(self.ncluster):
            mean, width, height, angle = normal.create_ellipse_patch_details(self.meansave[nlevel][cluster],
                self.Covsave[nlevel][cluster],self.weightsave[nlevel][cluster])
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                color=cm.jet((cluster+1)/self.ncluster), alpha=0.5)
            ax.add_patch(ell)

    def plot_cluster_animation(self,nlevel=-1,interval=500,title="",xlabel="",ylabel=""):
        fig,ax = plt.subplots(1,1)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel
        
        # list_object contains scatter plot for data and elliptical patch for weighted normal pdf for clusters
        list_object = [ax.scatter(self.X[0,:], self.X[1,:], color = cm.jet(0), marker="o", s=15)]
        for cluster in range(self.ncluster):
            # set up ellipse based on initial means, covariance matrices, and weights
            mean, width, height, angle = normal.create_ellipse_patch_details(self.meansave[0][cluster],
                self.Covsave[0][cluster],self.weightsave[0][cluster])
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                color=cm.jet((cluster+1)/self.ncluster), alpha=0.5)
            ax.add_patch(ell)
            list_object.append(ell)

        def update(frame_number,list_object,clustersave,meansave,Covsave,weightsave):
            nellipse = len(list_object)-1
            # update color of data points based on cluster assignments
            list_object[0].set_color(cm.jet((clustersave[frame_number]+1)/(nellipse)))
            # update elliptical patch details for clusters
            for cluster in range(nellipse):
                # call function to compute mean, width, height, angle for current frame 
                mean, width, height, angle = normal.create_ellipse_patch_details(meansave[frame_number][cluster],
                    Covsave[frame_number][cluster],weightsave[frame_number][cluster])
                # update center, width, height, and angle
                list_object[cluster+1].set_center(mean)
                list_object[cluster+1].width = width
                list_object[cluster+1].height = height
                list_object[cluster+1].angle = angle
            return list_object

        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[list_object,self.clustersave,self.meansave,self.Covsave,self.weightsave],
            repeat_delay=0, repeat=True, interval=interval, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('GMM_Animation.mp4', writer='ffmpeg')
        return ani