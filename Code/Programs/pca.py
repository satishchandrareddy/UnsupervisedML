# pca.py

import numpy as np

class pca:
    def __init__(self):
        pass

    def fit(self,X):
        # compute SVD of X = X - Xmean
        (self.dimension,self.nsample) = X.shape
        print("Number of data points: {}".format(self.nsample))
        print("Number of dimensions: {}".format(self.dimension))
        self.Xmean = np.mean(X,axis=1,keepdims=True)
        self.U,self.Sigma,self.Vh = np.linalg.svd(X-self.Xmean,full_matrices=False)
        self.cumulative_variance = np.cumsum(np.square(self.Sigma))/self.nsample
        self.total_variance = self.cumulative_variance[-1]
        print("Total Variance: {}".format(self.total_variance))
    
    def get_dimension(self,variance_capture):
        cumulative_variance_prop = self.cumulative_variance/self.total_variance
        cumulative_variance_capture = cumulative_variance_prop[cumulative_variance_prop<=variance_capture]
        dim = np.size(cumulative_variance_capture)
        if dim == 0:
            dim = 1
        else:
            if cumulative_variance_prop[dim-1]<variance_capture:
                dim += 1
        return dim

    def data_reduced_dimension(self,**kwargs):
        # compute coordinates of X-Xmean in reduced u(0),...,u(dim-1) coordinate system
        dim = self.dimension
        if "reduced_dim" in kwargs:
            dim = kwargs["reduced_dim"]
        elif "variance_capture" in kwargs:
            dim = self.get_dimension(kwargs["variance_capture"])
        print("Reduced dimension: {} - variance capture proportion: {}".format(dim,self.cumulative_variance[dim-1]/self.total_variance))
        return np.matmul(np.diag(self.Sigma[0:dim]),self.Vh[0:dim,:])

    def data_reconstructed(self,**kwargs):
        # recompute data in original number of dimensions using dim principal components
        dim = self.dimension
        if "reduced_dim" in kwargs:
            dim = kwargs["reduced_dim"]
        elif "variance_capture" in kwargs:
            dim = self.get_dimension(kwargs["variance_capture"])
        print("Reduced dimension: {} - variance capture proportion: {}".format(dim,self.cumulative_variance[dim-1]/self.total_variance))
        return np.matmul(self.U[:,0:dim],np.matmul(np.diag(self.Sigma[0:dim]),self.Vh[0:dim,:]))+self.Xmean