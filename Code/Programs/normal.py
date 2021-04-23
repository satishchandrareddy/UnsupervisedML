#normal.py

import numpy as np

def normal_pdf(X,mu,Cov):
	# Compute normal probability density function in arbitrary dimensions
	# X is np.array of shape (d dimensions x m data points) on which pdf is evaluated
	# mu is np.array of shape (d dimension x 1) of the mean
	# Cov is np.array of shape d,d - covariance matrix
    ndim,m = X.shape
    detCov = np.linalg.det(Cov)
    invCov = np.linalg.inv(Cov)
    Z = np.zeros((1,m))
    for count in range(m):
        prob = np.exp(-0.5*(np.dot((X[:,[count]]-mu).T,np.matmul(invCov,(X[:,[count]]-mu)))))
        Z[0,count] = prob/np.sqrt(np.power(2*np.pi,ndim)*detCov)
    # Z has dimensions (1, m data points)
    return Z

def normal_pdf_vectorized(X,mu,Cov):
	# Compute normal probability density function in arbitrary dimensions
	# X is 2d np.array of shape (d dimensions x m data points) on which pdf is evaluated
	# mu is 2d np.array of shape (d dimension x 1) of the mean
	# Cov is 2d np.array of shape d,d - covariance matrix
    detCov = np.linalg.det(Cov)
    invCov = np.linalg.inv(Cov)
    # compute inverse Cov * (X - mu) for all data points
    arg_exp = np.matmul(invCov,(X - mu))
    # compute -0.5* (X-mu)^T * [inverse Cov * (X-mu)] for all data points 
    arg_exp = -0.5*np.sum((X - mu)*arg_exp,axis=0,keepdims=True)
    # output is dimension (1, m data points)
    return np.exp(arg_exp)/np.sqrt(np.power(2*np.pi,X.shape[0])*detCov)

def create_ellipse_patch_details(mu,Cov,weight,contour=2e-3):
	# compute ellipse patch details in 2 dimensions to show contour where weighted pdf >= contour
	# if data is in dimension d>2, plots are for initial 2 dimensions only
	# mu (mean) is np.array of shape (d,) or np.array of shape (d,1)
	# Cov (covariance matrix) is np.array of shape (d,d)
	# weight (weight) is a float (real number)
	# contour is a float (real number)
	d = Cov.shape[1]
	detCov = np.linalg.det(Cov)
	alpha = np.sqrt(-2*np.log(contour*np.sqrt(np.power(2*np.pi,d)*detCov)/weight))
	U,Sigma,_ = np.linalg.svd(Cov)
	width = 2*alpha*np.sqrt(Sigma[0])
	height = 2*alpha*np.sqrt(Sigma[1])
	angle = np.arctan(U[1,0]/(U[0,0]+1e-10))*180/np.pi
	# return mean (only 2 components), width, height, angle
	return np.squeeze(mu)[0:2], width, height, angle