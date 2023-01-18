import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib import style
from matplotlib.colors import LogNorm
from copy import copy
#style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')

#Python/Data
import numpy as np
import uproot3 as ur #much easier array math than ur4
import awkward as ak

#For checking and making dirs
import os
import shutil
import inspect

#ML
import tensorflow as tf


class Strawman_Clusterer:
    def __init__(self,
                 file: str,
                 label: str,
                 detector_name: str,
                 sampling_fraction: float,
                 num_eventsMax = 100_000,
                 tree_name = 'events'
                 ):

        self.file = file
        self.label = label
        self.detector_name = detector_name
        self.sampling_fraction = sampling_fraction
        self.tree_name = tree_name

        self.hit_e_min = 0
        self.hit_t_max = 200
        self.cluster_e_min = 0

        self.path = "./"+label
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path)

        with ur.open(self.file) as ur_file:
            ur_tree = ur_file[self.tree_name]
            self.num_events = min(ur_tree.numentries, num_eventsMax)
            print(f"Loaded {self.num_events} Events")

    def run_clusterer(self):
            self.get_hits_e() #Ignor DeprecationWarning: `np.str`
            self.get_genP()
            self.get_cluster_sum()
            self.apply_cluster_cuts() #applies to cluster and genP
            self.apply_sampling_fraction()
            self.np_save_genP_clusterE()


    def get_hits_e(self):

        with ur.open(self.file) as ur_file:
            ur_tree = ur_file[self.tree_name]
            hits_e = ur_tree.array(f'{self.detector_name}.energy', entrystop=self.num_events)
            hits_t = ur_tree.array(f'{self.detector_name}.time', entrystop=self.num_events)
        
            #Min E and Max T Cuts on cells
            cuts = hits_e > self.hit_e_min                                                               
            cuts = np.logical_and( cuts, hits_t <= self.hit_t_max )
            self.hits_e = hits_e[cuts]

            #For QA, NOT for cluster sum!
            self.flat_hits_e = ak.ravel(self.hits_e[:,::10]) #only take every 10th hit

            # print(inspect.stack()[0][3]," Done") #prints current function
            return


    def get_cluster_sum(self):

        if self.hits_e_exist():

            print("Doing Cluster Sum...")
            self.cluster_sum = ak.to_numpy( ak.sum(self.hits_e,axis=-1) ) #takes a while...
            print("Cluster Sum Done!")

        return


    def get_genP(self):

        #particle at n=2 always has MC_GenStatus=1. We only need parent particle
        with ur.open(self.file) as ur_file:
            ur_tree = ur_file[self.tree_name]
            genPx = ur_tree.array('MCParticles.momentum.x',entrystop=self.num_events)[:,2]
            genPy = ur_tree.array('MCParticles.momentum.y',entrystop=self.num_events)[:,2]
            genPz = ur_tree.array('MCParticles.momentum.z',entrystop=self.num_events)[:,2]

        self.genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        self.genTheta = np.arccos(genPz/self.genP)*180/np.pi

        return


    def apply_cluster_cuts(self):
        #apply to data and label!

        if self.cluster_genP_exist():
            cluster_cut = self.cluster_sum > self.cluster_e_min
            self.cluster_sum = self.cluster_sum[cluster_cut]
            self.genP = self.genP[cluster_cut]

        return

    def apply_sampling_fraction(self):
        self.cluster_sum = self.cluster_sum/self.sampling_fraction
        print(f"Applied Sampling Fraction of {self.sampling_fraction} to Cluster Sums")

    def np_save_genP_clusterE(self):
        if self.cluster_genP_exist():
            np.save(f"{self.path}/flat_hits_e.npy",self.flat_hits_e)
            np.save(f"{self.path}/genP.npy",self.genP)
            np.save(f"{self.path}/genTheta.npy",self.genTheta)
            np.save(f"{self.path}/clusterSum.npy",self.cluster_sum)
            np.save(f"{self.path}/sampling_fraction.npy",self.sampling_fraction)
            print(f"Files saved to {self.path}/")

        return


    #Checking Functions
    def hits_e_exist(self):

        if not hasattr(self, 'hits_e'):
            print(f"Error: get_hits_e() needs to be run!")
            return False
        return True

        
    def cluster_genP_exist(self):

        self.hits_e_exist()
        if not hasattr(self, 'cluster_sum'):
            print(f"Error: get_cluster_sum() needs to be run first") 
            return False
        if not hasattr(self, 'genP'):
            print(f"Error: get_genP() needs to be run first") 
            return False
        return True
        

def load_ClusterSum_and_GenP(label):
    clusterSum = np.load(f"./{label}/clusterSum.npy")
    genP = np.load(f"./{label}/genP.npy")
    return clusterSum,genP

def load_flat_hits_e(label):
    flat_hits_e = np.load(f"./{label}/flat_hits_e.npy")
    return flat_hits_e

def energy_QA_plots(flat_hits_e, genP, cluster_sum, label):

    print("Plotting QA Distributions...")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 10), constrained_layout=True)
    axes = np.ravel(ax)

    max_hits_e = np.mean(flat_hits_e)+ 2*np.std(flat_hits_e)

    bins_hits_e = np.linspace(np.min(flat_hits_e),max_hits_e,100)
    axes[0].hist(flat_hits_e, bins=bins_hits_e, color="cyan", alpha=0.8)
    axes[0].set_ylabel("Counts",fontsize=22) 
    axes[0].set_xlabel("Cell Hit Energy [GeV]",fontsize=22) 
    axes[0].set_title("Cell Energy Distribution",fontsize=22) 

    axes[1].hist(np.ravel(genP),color="red",alpha=0.8)
    axes[1].set_ylabel("Counts",fontsize=22) 
    axes[1].set_xlabel("Generated Momentum [GeV]",fontsize=22) 
    axes[1].set_title("Gen. Momentum Distribution",fontsize=22) 

    axes[2].hist(cluster_sum,color="blue",alpha=0.8)
    axes[2].set_ylabel("Counts",fontsize=22) 
    axes[2].set_xlabel("Cluster Energy [GeV]",fontsize=22) 
    axes[2].set_title("Cluster Sum Distribution (Raw)",fontsize=22) 

    path = "./"+label
    plt.savefig(f"{path}/energy_QA_plots.pdf")

def ClusterSum_vs_GenP(clusterSum, genP, label, ylabel="Cluster Sum", plot_offset = 5.0):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), constrained_layout=True)
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))

    #Bins and Range
    sumE_maxPlot = 125.0 #max cluster energy to plot. Has high tails that mess up plotting
    sumE_maxPlot = min(sumE_maxPlot,np.max(clusterSum))
    cluster_bins = E_binning(np.min(clusterSum),sumE_maxPlot+plot_offset)
    truth_bins   = E_binning(np.min(genP),np.max(genP)+plot_offset)

    h, xedges, yedges = np.histogram2d(genP, clusterSum, bins=[truth_bins, cluster_bins])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=1.0e-2,vmax=1.1e4), rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=0)
    cb.ax.tick_params(labelsize=20)
    ax.set_xlabel("Generated Energy",fontsize=22)
    ax.set_ylabel("Cluster Sum Energy",fontsize=25)
    ax.set_title(f"Cluster Sum vs. Generated Energy",fontsize=30)

    draw_identity_line(ax, color='cyan', linewidth=2, alpha=0.5, label="Ideal")
    ax.legend(loc="upper left")
    fig.text(0.95,-0.05,label,transform=ax.transAxes,fontsize=10)


    path = "./"+label
    plt.savefig(f"{path}/ClusterSum_vs_GenP_Colormap.pdf") 

def E_binning(min_E, max_E,N_bins=100,log=False):

    bins = np.linspace(min_E,max_E,N_bins+1)

    if log:
        if min == 0:
            min = min+1
        if min < 0:
            print("Can't have negative energy and do logspace plotting!")
            return np.zeros(N_Bins+1)

        bins = np.logspace(min,max,N_bins+1)

    return bins

def draw_identity_line(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes








