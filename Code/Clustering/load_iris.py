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
	features = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.T
	print("Features.shape: {} - Labels.shape: {}".format(features.shape,labels.shape))
	return features, labels

if __name__ == "__main__":
	features, labels = load_iris()