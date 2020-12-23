#unittest_gaussianmm.py

import create_data_cluster_sklearn
import dbscan_sr
import numpy as np
import unittest

def test_label(minpts,epsilon,model):
    test_label = []
    test_border = [True]
    for idx in range(model.nsample):
        idx_neighbours = model.neighbours(model.X[:,[idx]])
        #print("list_label: {}".format(model.list_label[idx]))
        if model.list_label[idx] == "core":
            test_label.append(len(idx_neighbours)>=minpts)
        elif model.list_label[idx] == "border":
            test_label.append(len(idx_neighbours)<minpts)
            check = [model.list_label[count]=="core" for count in idx_neighbours]
            test_border.append(check)
        elif model.list_label[idx] == "noise":
            test_label.append(len(idx_neighbours)<minpts)
    final_check = all(test_label) and all(test_border)
    return final_check

class Test_functions(unittest.TestCase):
    
    def test_blobs(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"blobs")
        # create model
        minpts = 3
        epsilon = 0.4
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("blobs result: {}".format(test_result))
        self.assertTrue(test_result)
    
    def test_varied_blobs1(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"varied_blobs1")
        # create model
        minpts = 3
        epsilon = 0.35
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("varied_blobs1 result: {}".format(test_result))
        self.assertTrue(test_result)

    def test_varied_blobs2(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"varied_blobs2")
        # create model
        minpts = 4
        epsilon = 0.2
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("varied_blobs2 result: {}".format(test_result))
        self.assertTrue(test_result)

    def test_noisy_moons(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"noisy_moons")
        # create model
        minpts = 3
        epsilon = 0.2
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("noisy_moons result: {}".format(test_result))
        self.assertTrue(test_result)

    def test_noisy_circles(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"noisy_circles")
        # create model
        minpts = 7
        epsilon = 0.3
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("noisy_circles result: {}".format(test_result))
        self.assertTrue(test_result)

    def test_aniso(self):
        # generate data
        X = create_data_cluster_sklearn.create_data_cluster(1000,"aniso")
        # create model
        minpts = 8
        epsilon = 0.3
        model = dbscan_sr.dbscan_sr(minpts,epsilon)
        model.fit(X)
        test_result = test_label(minpts,epsilon,model)
        print("aniso result: {}".format(test_result))
        self.assertTrue(test_result)


if __name__ == "__main__":

    unittest.main()
