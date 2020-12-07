#unittest_normal.py

import normal
import numpy as np
import scipy.stats as stats
import unittest

class Test_functions(unittest.TestCase):
    
    def test_normal_1d(self):
        # create data
        variance = 0.9
        mu = 1.3
        X = np.random.randn(1,100)
        # from normal
        normal_pdf = normal.normal_pdf(X,np.array([[mu]]),np.array([[variance]]))
        # compare against scipy
        scipy_pdf = stats.multivariate_normal.pdf(X.T,mean=mu,cov=variance)
        # compute error and assert
        error = np.min(np.absolute(normal_pdf-scipy_pdf.T)/scipy_pdf.T)
        self.assertLessEqual(error,1e-10)

    def test_normal_2d(self):
        # create data
        covariance = np.array([[1.2,0.7],[0.7,1.4]])
        mu = np.array([[1.1],[-0.7]])
        X = np.random.randn(2,100)
         # from normal
        normal_pdf = normal.normal_pdf(X,mu,covariance)
        # compare against scipy
        scipy_pdf = stats.multivariate_normal.pdf(X.T,mean=mu[:,0],cov=covariance)
        # compute error and assert
        error = np.min(np.absolute(normal_pdf-scipy_pdf.T)/scipy_pdf.T)
        self.assertLessEqual(error,1e-10)

    def test_normal_3d(self):
        # create data
        covariance = np.array([[1.2,0.7,0.3],[0.7,2.1,0.1],[0.3,0.1,1.4]])
        mu = np.array([[1.1],[-0.7],[-0.3]])
        X = np.random.randn(3,100)
         # from normal
        normal_pdf = normal.normal_pdf(X,mu,covariance)
        # compare against scipy
        scipy_pdf = stats.multivariate_normal.pdf(X.T,mean=mu[:,0],cov=covariance)
        # compute error and assert
        error = np.min(np.absolute(normal_pdf-scipy_pdf.T)/scipy_pdf.T)
        self.assertLessEqual(error,1e-10)

if __name__ == "__main__":
    unittest.main()
