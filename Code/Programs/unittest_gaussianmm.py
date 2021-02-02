#unittest_gaussianmm.py

import create_data_cluster_sklearn
import gaussianmm
import numpy as np
from sklearn import mixture
import unittest

class Test_functions(unittest.TestCase):
    
    def test_1d_1cluster(self):
        # create data
        nfeature = 1
        nsample = 100
        ncluster = 1
        X = np.random.randn(nfeature,nsample)
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,10,1e-5,False)
        means = model.meansave[-1]
        covariances = model.Sigmasave[-1]
        weights = model.weightsave[-1]
        # results from sklearn
        model_sklearn = mixture.GaussianMixture(1)
        model_sklearn.fit(X.T)
        model_sklearn_means = model_sklearn.means_
        model_sklearn_weights = model_sklearn.weights_
        model_sklearn_covariances = model_sklearn.covariances_
        error_mean = np.sum(np.absolute(means-model_sklearn_means))
        error_weights = np.sum(np.absolute(weights-model_sklearn_weights))
        error_covariances = np.sum(np.absolute(covariances - model_sklearn_covariances))
        error = min([error_mean,error_weights,error_covariances])
        self.assertLessEqual(error,1e-10)

    def test_3d_1cluster(self):
        # create data
        nfeature = 3
        nsample = 300
        ncluster = 1
        X = np.random.randn(nfeature,nsample)
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,10,1e-5,False)
        means = model.meansave[-1]
        covariances = model.Sigmasave[-1]
        weights = model.weightsave[-1]
        # results from sklearn
        model_sklearn = mixture.GaussianMixture(1)
        model_sklearn.fit(X.T)
        model_sklearn_means = model_sklearn.means_
        model_sklearn_weights = model_sklearn.weights_
        model_sklearn_covariances = model_sklearn.covariances_
        error_mean = np.sum(np.absolute(means-model_sklearn_means))
        error_weights = np.sum(np.absolute(weights-model_sklearn_weights))
        error_covariances = np.sum(np.absolute(covariances - model_sklearn_covariances))
        error = min([error_mean,error_weights,error_covariances])
        self.assertLessEqual(error,1e-10)

    def test_2d_3cluster(self):
        np.random.seed(31)
        nsample = 1000
        case = "blobs"
        ncluster = 3
        X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
        nfeature = X.shape[0]
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,20,1e-5,False)
        means = model.meansave[-1]
        covariances = model.Sigmasave[-1]
        weights = model.weightsave[-1]
        # results from sklearn
        model_sklearn = mixture.GaussianMixture(ncluster)
        model_sklearn.fit(X.T)
        sklearn_means = model_sklearn.means_
        sklearn_weights = model_sklearn.weights_
        sklearn_covariances = model_sklearn.covariances_
        idx = [0,1,2]
        idx_sk = [2,0,1]
        diff_means = 0
        diff_cov = 0
        diff_weights = 0
        for count in range(ncluster):
            diff_means = np.max(np.absolute(means[idx[count]].T - sklearn_means[idx_sk[count]]))
            diff_cov = np.max(np.absolute(covariances[idx[count]] - sklearn_covariances[idx_sk[count]]))
            diff_weights = np.max(np.absolute(weights[idx[count]] - sklearn_weights[idx_sk[count]]))
        error = min([diff_means,diff_weights,diff_cov])
        self.assertLessEqual(error,1e-10)

if __name__ == "__main__":
    unittest.main()
