# dbscan.py

import utils
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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

  def expand_cluster(self, neighbours_i, i):
    """Expand the neighbours of current point recursively."""
    neighbours_i.remove(i)         # the point itself is part of its neighbours, remove it before expanding cluster
    neighbours_i = list(neighbours_i)

    while neighbours_i:                         
        j = neighbours_i.pop(0)                 # remove the first neighbour
        if self.labels_[j] == -1:                 # if a neighbour is labeled as noise before but part of i's neighbours then it's a border point
          self.labels_[j] = self.labels_[i]
          self.clusters_[self.labels_[i]].append(j)
        if self.labels_[j] is not None:           # point previously processed (already added to cluster), do nothing
          continue
        self.labels_[j] = self.labels_[i]           # add neighbour to cluster
        self.clusters_[self.labels_[i]].append(j)
        neighbours_j = self.find_neighbours(j)
        if len(neighbours_j) >= self.min_points:   # expand neighbours if j is a core point
          self.core_points_.append(j)             # mark j as a core point
          neighbours_i = neighbours_i + list(neighbours_j)

  def fit(self, X):
    """Train the model."""
    num_points = X.shape[1]
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
    
      # expand cluster recursively by all neighbours and neighbours of core points
      self.expand_cluster(neighbours_i, i)

  def set_params(self, **params):
    """Set the hyperparameters of the model."""
    for p in params:
      setattr(self, p, params[p])

  def get_params(self):
    """Get the hyperparameters of the model."""
    return {p: getattr(self, p) for p in ['epsilon', 'min_points', 'dist_func']}

  def plot_clusters(self, X):
    """Plot the final cluster assignments."""
    num_points = X.shape[1]
    fig,ax = plt.subplots()
    ax.set_title(f"DBSCAN - epsilon={self.epsilon}, minPoints={self.min_points}")
    plt.scatter(X[0,:], X[1,:], c=self.labels_)

  def plot_results_animation(self, X, notebook=False):
    """Create an animation of training process."""
    num_points = X.shape[1]
    labels_orig = np.array(self.labels_) + 1
    labels_current = ['b']*num_points
    indices = []
    fig,ax = plt.subplots()
    ax.set_title(f"DBSCAN - epsilon={self.epsilon}, minPoints={self.min_points}")
    frames = []
    # get the indices of observation in the order in which they were put in clusters
    for c in self.clusters_.values():
      indices = indices + c
    # create a different frame each time a point was added to a cluster
    for i in indices:
      labels_current[i] = utils.col_mapper[labels_orig[i]]
      frame = ax.scatter(X[0,:], X[1,:], c=labels_current.copy())
      frames.append([frame])
    ani = animation.ArtistAnimation(fig, frames, interval=50, repeat=False, blit=False) 
    # uncomment to create mp4 
    # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
    # ani.save('cluster.mp4', writer='ffmpeg')
    if notebook:
      return ani

    plt.show()
    
