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

    def test_normal_3d_vectorized(self):
        # create data
        covariance = np.array([[1.2,0.7,0.3],[0.7,2.1,0.1],[0.3,0.1,1.4]])
        mu = np.array([[1.1],[-0.7],[-0.3]])
        X = np.random.randn(3,100)
         # from normal
        normal_pdf = normal.normal_pdf_vectorized(X,mu,covariance)
        # compare against scipy
        scipy_pdf = stats.multivariate_normal.pdf(X.T,mean=mu[:,0],cov=covariance)
        # compute error and assert
        error = np.min(np.absolute(normal_pdf-scipy_pdf.T)/scipy_pdf.T)
        self.assertLessEqual(error,1e-10)        

    def test_ellipse_1(self):
        # create data
        mu = np.array([[0],[0]])
        covariance = np.array([[2.1,0.0],[0.0,1]])
        weight = 0.37
        contour = 5.3e-4
        mean, width, height, angle = normal.create_ellipse_patch_details(mu,covariance,weight,contour)
        X1 = np.array([[width/2],[0.0]])
        value1 = weight*normal.normal_pdf(X1,mu,covariance)
        error1 = np.sum(np.absolute(value1 - contour))
        X2 = np.array([[0.0],[height/2]])
        value2 = weight*normal.normal_pdf(X2,mu,covariance)
        error2 = np.sum(np.absolute(value2 - contour))
        error = np.maximum(error1,error2)
        self.assertLessEqual(error,1e-10)

    def test_ellipse_2(self):
         # create data
        mu = np.array([[1.1],[2.1]])
        covariance = np.array([[1.3,0.0],[0.0,2.3]])
        weight = 0.57
        contour = 1.2e-6
        mean, width, height, angle = normal.create_ellipse_patch_details(mu,covariance,weight,contour)
        X1 = mu+np.array([[height/2],[0.0]])
        value1 = weight*normal.normal_pdf(X1,mu,covariance)
        error1 = np.sum(np.absolute(value1 - contour))
        X2 = mu+np.array([[0.0],[width/2]])
        value2 = weight*normal.normal_pdf(X2,mu,covariance)
        error2 = np.sum(np.absolute(value2 - contour))
        error = np.maximum(error1,error2)
        self.assertLessEqual(error,1e-10)

    def test_ellipse_3_vectorized(self):
         # create data
        mu = np.array([[1.1],[2.1]])
        covariance = np.array([[1.3,0.7],[0.7,2.3]])
        weight = 0.57
        contour = 1.2e-6
        mean, width, height, angle = normal.create_ellipse_patch_details(mu,covariance,weight,contour)
        X1 = mu+0.5*width*np.array([[np.cos(angle*np.pi/180)],[np.sin(angle*np.pi/180)]])
        value1 = weight*normal.normal_pdf_vectorized(X1,mu,covariance)
        error1 = np.sum(np.absolute(value1 - contour))
        X2 = mu+0.5*height*np.array([[np.cos(angle*np.pi/180 + np.pi/2)],[np.sin(angle*np.pi/180 + np.pi/2)]])
        value2 = weight*normal.normal_pdf_vectorized(X2,mu,covariance)
        error2 = np.sum(np.absolute(value2 - contour))
        error = np.maximum(error1,error2)
        self.assertLessEqual(error,1e-10)


if __name__ == "__main__":
    unittest.main()
