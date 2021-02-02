#plot_datasets.py

import matplotlib.pyplot as plt
from matplotlib import cm
import create_data_cluster_sklearn

list_dataset = [["blobs", "varied_blobs1", "varied_blobs2"],["aniso", "noisy_moons", "noisy_circles"]]
nsample = 1500
nrow = 2
ncol = 3
fig, axs = plt.subplots(nrow,ncol)

for row in range(nrow):
	for col in range(ncol):
		case = list_dataset[row][col]
		X = create_data_cluster_sklearn.create_data_cluster(nsample,case)
		axs[row,col].scatter(X[0,:],X[1,:],color = cm.jet(0), s=15)
		axs[row,col].set_title(case)
plt.show()