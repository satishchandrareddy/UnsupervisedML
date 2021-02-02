#plot_datasets_iris.py

import load_iris
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
rcParams.update({"legend.fontsize": 6})

# load data
X,Y = load_iris.load_iris()
print(Y)
Ylabel = np.zeros(Y.shape)
list_label = ["setosa", "versicolor","virginica"]
# create integer labels
nclass = len(list_label)
for i,label in enumerate(list_label):
	print(label)
	list_idx = np.where(Y==label)[0]
	print(list_idx)
	Ylabel[list_idx] = i

# permutation of indices
list_feature = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
list_permutation = [[[0,1,2,3], [0,2,3,1], [0,3,1,2]],[[1,2,3,0], [1,3,0,2], [2,3,0,1]]]
nrow = 2
ncol = 3

# generate figure
fig, axs = plt.subplots(nrow,ncol,figsize=(10,7))
fig.suptitle("Iris Data")
for row in range(nrow):
    for col in range(ncol):
        permutation = list_permutation[row][col]
        Xnew = X[permutation,:]
        for count in range(nclass):
            idx = np.where(np.absolute(Ylabel - count)<1e-5)[0]
            axs[row,col].scatter(Xnew[0,idx],Xnew[1,idx],color = cm.hsv((count+1)/nclass), s=15, label=list_label[count])		    
        axs[row,col].set_xlabel(list_feature[list_permutation[row][col][0]])
        axs[row,col].set_ylabel(list_feature[list_permutation[row][col][1]])
        axs[row,col].legend(loc="upper left")
plt.show()