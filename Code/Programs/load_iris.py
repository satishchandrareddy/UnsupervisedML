# load_iris.py

import pandas as pd
from pathlib import Path

def load_iris():
	# read data from file
	root_dir = Path(__file__).resolve().parent.parent
	iris_df = pd.read_csv(root_dir / "Data_Iris/iris.csv")
	# extract labels
	labels = iris_df["species"].values
	# extract features
	features = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.T
	print("Number of dimensions: {} Number of data points: {}".format(features.shape[0],features.shape[1]))
	print("Number of labels: {}".format(labels.shape))
	return features, labels

if __name__ == "__main__":
	X, Y = load_iris()