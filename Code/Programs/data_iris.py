# data_iris.py

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pathlib import Path

class iris:
    def __init__(self):
        # store grandparent directory
        self.root_dir = Path(__file__).resolve().parent.parent

    def load(self):
        iris_df = pd.read_csv(self.root_dir / "Data_Iris/iris.csv")
        # extract labels
        class_label = iris_df["species"].values
        # save column headings and extract feature information
        self.feature = ["sepal_length","sepal_width","petal_length","petal_width"]
        X = iris_df[self.feature].values.T
        print("Number of dimensions: {} Number of data points: {}".format(X.shape[0],X.shape[1]))
        print("Number of class labels: {}".format(class_label.shape))
        return X, class_label

    def plot(self,X,class_label):
        rcParams.update({"legend.fontsize": 6})
        # get the class labels
        list_class_label = np.unique(class_label)
        nclass = len(list_class_label)
        # permutation of indices (variables)
        # 0 = sepal_length, 1 = sepal_width, 2 = petal_length, 3 = petal_width
        list_permutation = [[[0,1], [0,2], [0,3]],
                            [[1,2], [1,3], [2,3]]]
        # generate subplots 2 rows, 3 columns showing data for all possible combinations of 2 variables
        nrow = 2
        ncol = 3
        fig, ax = plt.subplots(nrow,ncol,figsize=(10,7))
        fig.suptitle("Iris Data")
        for row in range(nrow):
            for col in range(ncol):
                for i,classname in enumerate(list_class_label):
                    idx = np.where(class_label == classname)[0]
                    # scatter plot in x0-x1 plane where x0,x1 are entries 0,1 from list_permutation
                    ax[row,col].scatter(
                        X[list_permutation[row][col][0],idx],
                        X[list_permutation[row][col][1],idx],
                        color = cm.hsv((i+1)/nclass), s=15, label=classname)		    
                    ax[row,col].set_xlabel(self.feature[list_permutation[row][col][0]])
                    ax[row,col].set_ylabel(self.feature[list_permutation[row][col][1]])
                    ax[row,col].legend(loc="upper left")

if __name__ == "__main__":
	iris_object = iris()
	X,class_label = iris_object.load()
	iris_object.plot(X,class_label)
	plt.show()