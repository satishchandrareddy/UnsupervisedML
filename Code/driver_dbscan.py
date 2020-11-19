# driver_dbscan.py

import dbscan
import numpy as np
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

n_samples = 500
noisy_moons = ds.make_moons(n_samples=n_samples, noise=.08, random_state=0)
X = noisy_moons[0]
X = StandardScaler().fit_transform(X)

eps = 0.3
min_pts = 3
dist_func = lambda x1, x2: np.sqrt(np.dot(x1-x2,x1-x2))
model = dbscan.DBSCAN(epsilon=eps, 
					min_points=min_pts, 
					dist_func=dist_func)
model.fit(X)
labels = model.labels_
plt.scatter(X[:,0], X[:,1], c=labels)

print(f'Number of Clusters: {len(np.unique(labels))}')
print(f'Number of Core points: {len(model.core_points_)}')
print(f'Model Hyperparameters: {model.get_params()}')
plt.show()