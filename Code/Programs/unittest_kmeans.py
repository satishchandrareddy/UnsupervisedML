#unittest_kmeans.py

import kmeans
import numpy as np
import unittest

def dist(X,mean):
    dist = np.zeros((len(mean),X.shape[1]))
    for count in range(len(mean)):
        dist[count,:] = np.sum(np.square(X-mean[count]),axis=0,keepdims=True)
    return dist

def kmeans_test(X,clustersave,meansave):
    bool_test = []
    nlevel = len(meansave)
    for level in range(1,nlevel):
        dist2 = dist(X,meansave[level-1])
        cluster_assignment = np.argmin(dist2,axis=0)
        bool_test_level = (cluster_assignment == clustersave[level])
        bool_test.append(bool_test_level.all())
    return all(bool_test)
        
class Test_functions(unittest.TestCase):
            
    def test_3cluster(self):
        # create data
        nfeature = 5
        nsample = 1000
        X = np.random.randn(nfeature,nsample)
        # create model and fit
        ncluster = 3
        model = kmeans.kmeans(ncluster)
        model.fit(X,20,1e-3,False)
        # test result
        bool_test = kmeans_test(X,model.clustersave,model.meansave)
        self.assertTrue(bool_test)

if __name__ == "__main__":
    unittest.main()
