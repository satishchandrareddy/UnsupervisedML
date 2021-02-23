#normal.py

import numpy as np

def normal_pdf(X,mu,Cov):
	# Compute normal probability density function in arbitrary dimensions
	# X is np.array of shape (d dimensions x m data points) on which pdf is evaluated
	# mu is np.array of shape d dimension x 1) of the mean
	# Cov is np.array of shape d,d - covariance matrix
    nfeature,nsample = X.shape
    detCov = np.linalg.det(Cov)
    invCov = np.linalg.inv(Cov)
    output = np.zeros((1,nsample))
    for count in range(nsample):
        prob = np.exp(-0.5*(np.dot((X[:,[count]]-mu).T,np.dot(invCov,(X[:,[count]]-mu)))))
        output[0,count] = prob/np.sqrt(np.power(2*np.pi,nfeature)*detCov)
    # output is dimension (1, m data points)
    return output

def normal_pdf_vectorized(X,mu,Cov):
	# Compute normal probability density function in arbitrary dimensions
	# X is 2d np.array of shape (d dimensions x m data points) on which pdf is evaluated
	# mu is 2d np.array of shape (d dimension x 1) of the mean
	# Cov is 2d np.array of shape d,d - covariance matrix
    detCov = np.linalg.det(Cov)
    invCov = np.linalg.inv(Cov)
    arg_exp = np.matmul(invCov,(X - mu))
    arg_exp = -0.5*np.sum((X - mu)*arg_exp,axis=0,keepdims=True)
    # output is dimension (1, m data points)
    return np.exp(arg_exp)/np.sqrt(np.power(2*np.pi,X.shape[0])*detCov)

def create_ellipse_patch_details(mu,Cov,weight,contour=2e-3):
	# compute ellipse patch in 2d to show contour where normal pdf <= contour
	# mu (mean) is np.array of shape (2,) or np.array of shape (2,1)
	# Cov (covariance matrix) is np.array of shape (2,2)
	# weight (weight) is a real number
	# contour is a number
	detCov = np.linalg.det(Cov)
	alpha = np.sqrt(-2*np.log(contour*2*np.pi*np.sqrt(detCov)/weight))
	U,Sigma,_ = np.linalg.svd(Cov)
	width = 2*alpha*np.sqrt(Sigma[0])
	height = 2*alpha*np.sqrt(Sigma[1])
	angle = np.arctan(U[0,1]/(U[0,0]+1e-10))*180/np.pi
	# return mean, width, height, angle
	return np.squeeze(mu), width, height, angle