# load_iris.py

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pathlib import Path

class iris:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def load(self):
        iris_df = pd.read_csv(self.root_dir / "Data_Iris/iris.csv")
        # extract labels
        labels = iris_df["species"].values
        # extract features
        self.list_feature = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        features = iris_df[self.list_feature].values.T
        print("Number of dimensions: {} Number of data points: {}".format(features.shape[0],features.shape[1]))
        print("Number of labels: {}".format(labels.shape))
        return features, labels

    def plot(self,X,Y):
        rcParams.update({"legend.fontsize": 6})
        # labels
        nclass = 3
        dict_label = {"setosa": 0, "versicolor": 1, "virginica": 2}
        # permutation of indices
        list_permutation = [[[0,1,2,3], [0,2,3,1], [0,3,1,2]],[[1,2,3,0], [1,3,0,2], [2,3,0,1]]]
        # generate figure showing data for all possible combinations of 2 features
        nrow = 2
        ncol = 3
        fig, axs = plt.subplots(nrow,ncol,figsize=(10,7))
        fig.suptitle("Iris Data")
        for row in range(nrow):
            for col in range(ncol):
                permutation = list_permutation[row][col]
                Xnew = X[permutation,:]
                for label in dict_label.keys():
                    idx = np.where(Y == label)[0]
                    axs[row,col].scatter(Xnew[0,idx],Xnew[1,idx],color = cm.hsv((dict_label[label]+1)/nclass), s=15, label=label)		    
                    axs[row,col].set_xlabel(self.list_feature[list_permutation[row][col][0]])
                    axs[row,col].set_ylabel(self.list_feature[list_permutation[row][col][1]])
                    axs[row,col].legend(loc="upper left")

if __name__ == "__main__":
	iris_object = iris()
	X,Y = iris_object.load()
	iris_object.plot(X,Y)
	plt.show()