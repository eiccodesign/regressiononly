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
                 n_Z_layers=3, #start with 3
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

        self.n_Z_layers=n_Z_layers
        
        self.take_log = take_log

        self.path = "./"+label
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path)

        with ur.open(self.file) as ur_file:
            ur_tree = ur_file[self.tree_name]
            self.num_events = min(ur_tree.numentries, num_eventsMax)
            print(f"Loaded {self.num_events} Events")

    def run_clusterer(self):
            self.get_genP()
            self.get_hits_e() #Ignor DeprecationWarning: `np.str`
            self.get_cluster_sum()
            self.apply_cluster_cuts() #applies to cluster and genP
            self.apply_sampling_fraction()
            self.np_save_genP_clusterE()


    def run_segmentation_clusterer(self):
            self.get_genP()
            self.get_hits_e()
            self.get_hits_z()
            self.get_segmented_cluster_sum()
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
            self.hit_cuts = cuts
            self.hits_e = hits_e[cuts]

            #For QA, NOT for cluster sum!
            self.flat_hits_e = ak.ravel(self.hits_e[:,::10]) #only take every 10th hit

            # print(inspect.stack()[0][3]," Done") #prints current function
            return

    def get_hits_z(self):

        if self.hits_e_exist():
            print("Getting Cell Z information")

            with ur.open(self.file) as ur_file:
                ur_tree = ur_file[self.tree_name]
                hits_z = ur_tree.array(f'{self.detector_name}.position.z', entrystop=self.num_events)
            
                #Min E and Max T Cuts on cells, obtained from get_hits_e()
                self.hits_z = hits_z[self.hit_cuts]
    
                #For QA
                self.flat_hits_z = ak.ravel(self.hits_z[:,::10]) #only take every 10th hit

        return

    def get_cluster_sum(self):

        if self.hits_e_exist():
            print("Doing Cluster Sum...")

            self.cluster_sum = ak.to_numpy( ak.sum(self.hits_e,axis=-1) ) #takes a while...

            if (self.take_log):
                self.cluster_sum = np.log(self.cluster_sum)
            print("Cluster Sum Done!")

        return

    def get_segmented_cluster_sum(self):

        if self.hits_z_exist():

            z_layers = get_Z_segmentation(self.hits_z,self.n_Z_layers)
            self.z_mask =  get_Z_masks(self.hits_z,z_layers)

            cluster_sum = []
            for zbin in range(self.n_Z_layers):
                mask = self.z_mask[zbin]
                print("Doing Cluster Sum for z-bin %i..."%(zbin))
                cluster_sum.append(ak.to_numpy(ak.sum(self.hits_e[mask] ,axis=-1)))

            self.segmented_cluster_sum = np.asarray(cluster_sum)
            self.cluster_sum = np.sum(self.cluster_sum, axis=0)
            
            if (self.take_log):
                self.cluster_sum = np.log(self.cluster_sum)
            print("Cluster Sum(s) Done!")

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
        ''' apply to data and label! '''

        cluster_cut = self.cluster_sum > self.cluster_e_min
        self.cluster_sum = self.cluster_sum[cluster_cut]

        if self.cluster_genP_exist():
            self.genP = self.genP[cluster_cut]


        if self.segmented_cluster_sum_exist():

            if (np.shape(self.cluster_sum)[0] == self.n_Z_layers) :

                print("Before Cuts, N = ",np.shape(self.cluster_sum))
                new_clusters = []
                cluster_cut = np.sum(self.cluster_sum, axis=0) > self.cluster_e_min
                print(np.shape(cluster_cut))

                for z_bin in range(self.n_Z_layers):
                    cluster_segment = self.cluster_sum[z_bin]
                    new_clusters.append(cluster_segment[cluster_cut])

                self.cluster_sum = np.asarray(new_clusters)
                print("After Cuts, N = ",np.shape(self.cluster_sum))



        return

    def apply_sampling_fraction(self):
        self.cluster_sum = self.cluster_sum/self.sampling_fraction
        print(f"Applied Sampling Fraction of {self.sampling_fraction} to Cluster Sums")

    def np_save_genP_clusterE(self):

        np.save(f"{self.path}/clusterSum.npy",self.cluster_sum)
        print(f"cluster sum saved to {self.path}/clusterSum.npy")

        if self.segmented_cluster_sum_exist():
            save_npy(self.segmented_cluster_sum, "segmented_clusterSum")

        if self.cluster_genP_exist():
            save_npy(self.genP, "genP")
            save_npy(self.genTheta, "genTheta")

        if (self.hits_e_exist):
            save_npy(self.hits_e, "flat_hits_e")

        if (self.hits_z_exist):
            save_npy(self.hits_z, "flat_hits_z")
       
        save_npy(self.sampling_fraction, "sampling_fraction")

        print(f"Files saved to {self.path}/")

        return


    #Checking Functions
    def hits_e_exist(self):

        if not hasattr(self, 'hits_e'):
            print(f"Error: get_hits_e() needs to be run!")
            return False
        return True

    def hits_z_exist(self):

        if not hasattr(self, 'hits_z'):
            print(f"Error: get_hits_z() needs to be run!")
            return False
        return True

        
    def cluster_genP_exist(self):

        self.hits_e_exist()
        
        if not hasattr(self, 'cluster_sum'):
            print(f"Error: genP not set. get_cluster_sum() needs to be run first") 
            return False
        if not hasattr(self, 'genP'):
            print(f"Error: get_genP() needs to be run") 
            return False
        return True
        
    def segmented_cluster_sum_exist(self):

        self.hits_e_exist()

        if not hasattr(self, 'segmented_cluster_sum'):
            print(f"segmented cluster sum not calculated. (OK if doing 1-1 regression) ")
            return False

    def save_npy(self, data, name):
        np.save(f"{self.path}/{name}.npy",data)
        print(f" {data} saved to {self.path}/{data}.npy")

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

def get_Z_segmentation(cell_z_array, n_Z_Bins):

    zmin = ak.min(cell_z_array)
    zmax = ak.max(cell_z_array)

    return np.linspace(zmin,zmax,n_Z_Bins +1)

def get_Z_masks(cell_z_array,z_layers):

    zmask = []
    for i in range(len(z_layers)-1):
        zmask.append(np.logical_and(cell_z_array >= z_layers[i], cell_z_array < z_layers[i+1]))

    return zmask

