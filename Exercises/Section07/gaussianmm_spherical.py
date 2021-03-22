# gaussianmm_spherical.py
# Constant variance version of the Gaussian Mixture Model

import gaussianmm
import numpy as np

def normal_spherical(X,mu,var):
    # Compute normal probability density function in arbitrary dimensions
    # X is np.array of shape (d dimensions x m data points) on which pdf is evaluated
    # mu is np.array of shape d dimension x 1) of the mean
    # var is variance
    nfeature,nsample = X.shape
    prob = np.exp(-0.5*np.sum((X - mu)*(X - mu),axis=0,keepdims=True)/var)
    return prob/np.power(2*np.pi*var,nfeature/2)

class gaussianmm(gaussianmm.gaussianmm):
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
            array_idx = np.random.choice(self.X.shape[1],self.ncluster)
            mean = [self.X[:,[array_idx[count]]] for count in range(self.ncluster)]
        # meansave is list of list of means
        self.meansave = [mean]
        # initialize weights = 1/# of clusters for all gaussians
        self.weightsave = [[1/self.ncluster for _ in range(self.ncluster)]]
        # ***CHANGE for sphere
        # compute initial variance
        Xmm = self.X - np.mean(self.X,axis=1,keepdims=True)
        variance = np.sum(Xmm*Xmm)/self.nsample/Xmm.shape[0]
        # save current variance for each component
        self.variance = [variance for _ in range(self.ncluster)]
        # create diagonal covariance matrix for each component
        self.Covsave = [[np.diag(self.variance[k]*np.ones((Xmm.shape[0]))) for k in range(self.ncluster)]]

    def expectation(self):
        # expectation step - update conditional probabilities
        weighted_normal = np.zeros((self.ncluster,self.nsample))
        for k in range(self.ncluster):
            # *** CHANGE for sphere: use normal_spherical function for normal pdf
            weighted_normal[k,:] = self.weightsave[-1][k]*normal_spherical(self.X,self.meansave[-1][k],self.variance[k])
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
            # *** CHANGE for sphere: compute variance for component then create constant diag covariance 
            self.variance[k] = np.sum(self.gamma[k,:]*Xmm*Xmm)/self.M[k]/Xmm.shape[0]
            list_Cov.append(np.diag(self.variance[k]*np.ones((Xmm.shape[0]))))
        self.meansave.append(list_mean)
        self.weightsave.append(list_weight)
        self.Covsave.append(list_Cov)