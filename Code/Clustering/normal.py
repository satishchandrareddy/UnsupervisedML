#normal.py

import numpy as np

def normal_pdf(X,mu,Sigma):
    nfeature,nsample = X.shape
    detSigma = np.linalg.det(Sigma)
    invSigma = np.linalg.inv(Sigma)
    output = np.zeros((1,nsample))
    for count in range(nsample):
        prob = np.exp(-0.5*(np.dot((X[:,count:count+1]-mu).T,np.dot(invSigma,(X[:,count:count+1]-mu)))))
        output[0,count] = prob/np.sqrt(np.power(2*np.pi,nfeature)*detSigma)
    return output