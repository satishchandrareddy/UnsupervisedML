#unittest_gaussianmm.py

import create_data_cluster
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
        X,_ = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster)
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,10,False)
        means = model.get_meansave()[-1]
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
        X,_ = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster)
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,10,False)
        means = model.get_meansave()[-1]
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

'''
    def test_1d_2cluster(self):
        # create data
        nfeature = 1
        nsample = 5
        ncluster = 2
        X,_ = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster)
        # results from gaussianmm
        model = gaussianmm.gaussianmm(ncluster)
        model.fit(X,20,False)
        means = model.get_meansave()[-1]
        covariances = model.Sigmasave[-1]
        weights = model.weightsave[-1]
        print("mean: {}".format(means))
        print("covariances: {}".format(covariances))
        print("weights: {}".format(weights))
        # results from sklearn
        model_sklearn = mixture.GaussianMixture(ncluster)
        model_sklearn.fit(X.T)
        sklearn_means = model_sklearn.means_
        sklearn_weights = model_sklearn.weights_
        sklearn_covariances = model_sklearn.covariances_
        print("sklearn_mean: {}".format(sklearn_means))
        print("sklearn_covariances: {}".format(sklearn_covariances))
        print("sklearn_weights: {}".format(sklearn_weights))
        error_mean = np.sum(np.absolute(means-sklearn_means))
        error_weights = np.sum(np.absolute(weights-sklearn_weights))
        error_covariances = np.sum(np.absolute(covariances - sklearn_covariances))
        error = min([error_mean,error_weights,error_covariances])
        self.assertLessEqual(error,1e-10)
'''

if __name__ == "__main__":
    unittest.main()
