# clustering_base.py

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class clustering_base:
    def __init__(self):
        pass

    def initialize_algorithm(self):
        pass

    def fit(self,X):
        pass

    def get_index(self,nlevel,cluster_number):
        # nlevel (integer)
        # cluster_number (integer)
        # return indices of samples where clustersave[nlevel] = cluster_number
        return np.where(np.absolute(self.clustersave[nlevel]-cluster_number)<1e-5)[0]

    def plot_objective(self,title="",xlabel="",ylabel=""):
        # plot objective function if data is collected
        if len(self.objectivesave)>0:
            fig = plt.figure()
            plt.plot(self.objectivesave,'b-',marker="o",markersize=5)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    def plot_cluster(self,nlevel=-1,title="",xlabel="",ylabel=""):
        # plot cluster assignment for dataset for self.clustersave[nlevel]
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        color = (self.clustersave[nlevel]+1)/self.ncluster
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(color),marker="o",s=15)

    def plot_cluster_animation(self,nlevel=-1,interval=50,title="",xlabel="",ylabel=""):
        # create animation for cluster assignments in self.clustersave[level] 
        # for level = 0,1,...,nlevel
        # interval is the time (in milliseconds) between frames in animation 
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel
        # scatter plot all data points in same color
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)
        # update function for animation change color according to cluster assignment
        def update(i,scat,clustersave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat.set_color(cm.jet(array_color_data))
            return scat,
        # create animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[scat,self.clustersave,self.ncluster], repeat_delay=5000, repeat=True, interval=interval, blit=True)
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg on internet to get detaisl
        #ani.save('Clustering_Animation.mp4', writer='ffmpeg')
        return ani