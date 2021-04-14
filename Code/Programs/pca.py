# pca.py

import matplotlib.pyplot as plt
import numpy as np

class pca:
    def __init__(self):
        pass

    def fit(self,X):
        # compute compact SVD of X - Xmean
        self.dimension,self.nsample = X.shape
        print("Number of dimensions: {}  data points: {}".format(self.dimension,self.nsample))
        self.Xmean = np.mean(X,axis=1,keepdims=True)
        self.U,self.Sigma,self.Vt = np.linalg.svd(X-self.Xmean,full_matrices=False)
        cumulative_variance = np.cumsum(np.square(self.Sigma))/(self.nsample-1)
        self.cumulative_variance_proportion = cumulative_variance/cumulative_variance[-1]
    
    def get_dimension(self,variance_capture):
        # variance capture (float): proportion of variance to be captured 0<variance_capture<=1
        # return: minimum number of dimensions to capture at least variance_capture proportion of variance
        cumulative_variance_capture = np.where(self.cumulative_variance_proportion < variance_capture)[0]
        return np.size(cumulative_variance_capture)+1

    def data_reduced_dimension(self,**kwargs):
        # *kwargs input one of 
        # reduced_dim (integer) reduced dimension
        # variance_capture (float) is amount of variance to be captured 0<variance_capture<=1
        # return reduced dimension version of X-Xmean in u(0),...,u(K-1) coordinate system
        K = self.dimension
        if "reduced_dim" in kwargs:
            K = kwargs["reduced_dim"]
        elif "variance_capture" in kwargs:
            K = self.get_dimension(kwargs["variance_capture"])
        print("Reduced dimension: {} - variance capture proportion: {}".format(K,self.cumulative_variance_proportion[K-1]))
        return np.matmul(np.diag(self.Sigma[0:K]),self.Vt[0:K,:])

    def data_reconstructed(self,**kwargs):
        # *kwargs input one of 
        # reduced_dim (integer) reduced dimension
        # variance_capture (float) is amount of variance to be captured 0<variance_capture<=1
        # return reconstucted dataset in original coordinate system using reduced number of dimensions
        K = self.dimension
        if "reduced_dim" in kwargs:
            K = kwargs["reduced_dim"]
        elif "variance_capture" in kwargs:
            K = self.get_dimension(kwargs["variance_capture"])
        print("Reduced dimension: {} - variance capture proportion: {}".format(K,self.cumulative_variance_proportion[K-1]))
        return np.matmul(self.U[:,0:K],np.matmul(np.diag(self.Sigma[0:K]),self.Vt[0:K,:]))+self.Xmean
    
    def plot_cumulative_variance_proportion(self):
        # plot cumulative_variance_proportion
        plt.figure()
        principal_components = np.arange(1,np.size(self.cumulative_variance_proportion)+1)
        plt.plot(principal_components,self.cumulative_variance_proportion)
        plt.title("Cumulative Variance Proportion")
        plt.xlabel("Principal Components")
        plt.ylabel("Proportion")

if __name__ == "__main__":
    #(1) dataset
    X = np.array([[4,-7,-3,4],[8,5,6,5],[9,1,-1,3]])
    print("X: \n{}".format(X))
    #(2) create instance of pca class
    model = pca()
    #(3) compute compact svd
    model.fit(X)
    print("Cumulative Variance Proportion: {}".format(model.cumulative_variance_proportion))
    model.plot_cumulative_variance_proportion()
    #(4) compute dataset in 2 dimensions
    Ra = model.data_reduced_dimension(reduced_dim=2)
    print("R: \n{}".format(Ra))
    Xa = model.data_reconstructed(reduced_dim=2)
    print("XR Reconstructed: \n{}".format(Xa))
    print("---------------------------------------------------------------------------")
    #(4) compute dataset capturing 80% of variance
    Rb = model.data_reduced_dimension(variance_capture=0.80)
    print("R: \n{}".format(Rb))
    Xb = model.data_reconstructed(variance_capture=0.80)
    print("XR Reconstructed: \n{}".format(Xb))
    plt.show()