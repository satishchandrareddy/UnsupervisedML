# dbscan.py

import utils

class DBSCAN:
  def __init__(self, epsilon, min_points, dist_func):
    self.epsilon = epsilon
    self.min_points = min_points
    self.dist_func = dist_func

  def find_neighbours(self, p):
    """Find the set all neighbours of a point that are epsilon away from it, 
      including itself.
    """
    neighbours = set()
    for i in range(len(self.dist_matrix_)):
      if self.dist_matrix_[p, i] <= self.epsilon:
        neighbours.add(i)
    return neighbours

  def expand_cluster(self, neighbours_i, cluster_label):
    """Expand the neighbours of current point (i) by all neighbours 
      of its neighbours.
    """
    while neighbours_i:                         
        j = neighbours_i.pop()
        if self.labels_[j] == -1:                 # if one of the neighbours labeled as noise before but part of i's neighbours then it's a border point
          self.labels_[j] = cluster_label
          self.clusters_[cluster_label].append(j)
        if self.labels_[j] is not None:           # point previously processed (already added to cluster), do nothing
          continue
        self.labels_[j] = cluster_label           # add neighbour to cluster
        self.clusters_[cluster_label].append(j)
        neighbours_j = self.find_neighbours(j)
        if len(neighbours_j) >= self.min_points:   # expand neighbours
          neighbours_i = neighbours_i.union(neighbours_j)
          self.core_points_.append(j)

  def fit(self, X):
    """Train the model."""
    num_points = len(X)
    self.dist_matrix_ = utils.distance_matrix(X, self.dist_func)    # compute pairwise distance between each point
    self.core_points_  = []                   # store indices of core points
    self.labels_ = [None]*num_points          # each entry i in labels_ is the cluster label assigned to point i in dataset
    self.clusters_ = {}                     # clusters_[i] is a list of indices of observations assigned to cluster i
    cluster_label = -1

    for i in range(num_points):
      if self.labels_[i] is not None:         # point already visited, do nothing
        continue
      neighbours_i = self.find_neighbours(i)    # find neighbours of current point
      if len(neighbours_i) < self.min_points:     # check density
        self.labels_[i] = -1                      # label as noise
        continue
      self.core_points_.append(i)                 # i is a core point since |neighbours_i| >= min_points
      cluster_label += 1
      self.clusters_[cluster_label] = [i]         # add current point to current cluster
      self.labels_[i] = cluster_label
      neighbours_i.remove(i)                      # the point itself is part of its neighbours, remove it before expanding cluster
    
      # expand cluster recursively by all neighbours of current point's neighbours
      self.expand_cluster(neighbours_i, cluster_label)

  def set_params(self, **params):
    """Set the hyperparameters of the model."""
    for p in params:
      setattr(self, p, params[p])

  def get_params(self):
    """Get the hyperparameters of the model."""
    return {p: getattr(self, p) for p in ['epsilon', 'min_points', 'dist_func']}
      