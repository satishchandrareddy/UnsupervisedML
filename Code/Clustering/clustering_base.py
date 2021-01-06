# clustering_base.py

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class clustering_base:
    def __init__(self):
        pass

    def initialize_parameters(self):
        pass

    def fit(self,X):
        pass

    def plot_cluster(self,nlevel=-1,title="",xlabel="",ylabel=""):
        # 
        fig,ax = plt.subplots(1,1)
        # plot data points separate color for each cluster
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        color = (self.clustersave[nlevel]+1)/self.ncluster
        clusterdata = plt.scatter(self.X[0,:],self.X[1,:],color=cm.jet(color),marker="o",s=15)

    def plot_animation(self,nlevel=-1,title="",xlabel="",ylabel=""):
        fig,ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        nframe = len(self.clustersave)
        if nlevel < 0:
            nframe = nframe + 1 + nlevel
        else:
            nframe = nlevel
        scat = ax.scatter(self.X[0,:],self.X[1,:],color=cm.jet(0),marker="o",s=15)

        def update(i,scat,clustersave,ncluster):
            array_color_data = (1+self.clustersave[i])/(self.ncluster+1e-16)
            scat.set_color(cm.jet(array_color_data))
            return scat,

        ani = animation.FuncAnimation(fig=fig, func=update, frames = nframe,
            fargs=[scat,self.clustersave,self.ncluster], repeat_delay=1000, repeat=True, interval=50, blit=True)