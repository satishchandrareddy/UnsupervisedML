# utils.py

import numpy as np

def distance_matrix(X, dist_func):
  """Compute distance matrix such that entry i,j is the 
    distance between point i and point j
  """
  num_points = X.shape[1]
  D = np.zeros((num_points, num_points))

  for i in range(num_points):
    for j in range(num_points):
      if i != j:
        D[i,j] = dist_func(X[:,i], X[:,j])

  return D