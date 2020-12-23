#normal.py

import numpy as np

def normal_pdf(X,mu,Cov):
	# Compute normal probability density function in arbitrary dimensions
	# X is np.array of shape d dimensions, m data points on which pdf is evaluated
	# mu is np.array of shape d dimension,1 of the mean
	# Cov is np.array of shape d,d - covariance matrix
    nfeature,nsample = X.shape
    detCov = np.linalg.det(Cov)
    invCov = np.linalg.inv(Cov)
    output = np.zeros((1,nsample))
    for count in range(nsample):
        prob = np.exp(-0.5*(np.dot((X[:,count:count+1]-mu).T,np.dot(invCov,(X[:,count:count+1]-mu)))))
        output[0,count] = prob/np.sqrt(np.power(2*np.pi,nfeature)*detCov)
    return output

def create_ellipse_patch_details(mu,Cov,weight,contour=1e-8):
	# compute ellipse patch in 2d to show contour 
	# mu (mean) is np.array of shape (2,) or np.array of shape (2,1)
	# Cov (covariance matrix) is np.array of shape (2,2)
	# weight (weight) is a scalar
	detCov = np.linalg.det(Cov)
	alpha = np.sqrt(-2*np.log(contour*2*np.pi*np.sqrt(detCov)/weight))
	U,Sigma,_ = np.linalg.svd(Cov)
	width = alpha*np.sqrt(Sigma[0])
	height = alpha*np.sqrt(Sigma[1])
	angle = np.arctan(U[0,1]/(U[0,0]+1e-10))*180/np.pi
	# return mean, width, height, angle
	return np.squeeze(mu), width, height, angle