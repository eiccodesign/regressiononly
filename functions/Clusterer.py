import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib import style
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
                 tree_name = 'events',
                 take_log = False
                 ):

        self.file = file
        self.label = label
        self.detector_name = detector_name
        self.sampling_fraction = sampling_fraction
        self.tree_name = tree_name

        self.hit_e_min = 0
        self.hit_t_max = 200
        self.cluster_e_min = 0
        
        self.take_log = take_log

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
            if (self.take_log):
                self.cluster_sum = np.log(self.cluster_sum)
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

        if (self.take_log):
            self.genP = np.log(self.genP)

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

