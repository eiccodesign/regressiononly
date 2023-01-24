#Processing
import numpy as np
import uproot as ur
import awkward as ak

#ML
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler

#Plotting
import matplotlib.pyplot as plt

import numpy as np
import uproot as ur
import awkward as ak
import os
import time
from multiprocessing import Process, Queue, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
import random

#Simple function for defining the data generator (interface to data as if it were an infinite stream)
#Mean and standard deviations are obtained from training dataset for performing a standard scalar transformation

scales = {
    'cell_E_mean': 0.0,
    'cell_E_std': 0.0,
    'cell_X_mean': 0.0,
    'cell_X_std': 0.0,
    'cell_Y_mean': 0.0,
    'cell_Y_std': 0.0,
    'cell_Z_mean': 0.0,
    'cell_Z_std': 0.0}

class generator:
    def __init__(self, file, dataset):
        self.file = file
        self.dataset = dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for data in hf[self.dataset]:
                yield data

class data_generator:
    def __init__(self, file, input_dataset, target_dataset,batch_size=1000,
                 do_norm=True,path="./",get_scalar=True,n_scalar_batches=100):

        self.file = file
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.do_norm = do_norm
        self.path = path

        self.input_means = input_means
        self.input_stdevs = input_stdevs
        self.target_means = target_means
        self.target_stdevs = target_stdevs


class data_generator:
    def __init__(self,
                 file_list: list,
                 cellGeo_file: str,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs = 32,
                 preprocess = False,
                 output_dir = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir

        if self.preprocess and self.output_dir is not None:
            self.file_list = file_list
            self.num_files = len(self.file_list)

        else:
            self.file_list = file_list
            self.num_files = len(self.file_list)

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.num_procs = num_procs
        self.procs = []

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

        self.detector_name = "HcalEndcapPHitsReco" #'Insert' after the 'P'
        self.sampling_fraction = 0.02 #0.0098 for insert


    def preprocess_data(self):
        print('\nPreprocessing and saving data to {}'.format(self.output_dir))
        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
        
        for p in self.procs:
            p.join()

        self.file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]

    def preprocessor(self, worker_id):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Proceesing file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            preprocessed_data = []

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]
                
                # cluster_cells_X = event_data[self.detector_name+".position.x"]
                # cluster_cells_Y = event_data[self.detector_name+".position.y"]
                # cluster_cells_Z = event_data[self.detector_name+".position.z"]
                #FIXME: for pfn and GNNs after first test

                cluster_cells_E = event_data[self.detector_name+".energy"]
                cluster_sum_E = np.sum(cluster_cells_E,axis=-1) #global node feature later
                cluster_calib_E = get_cluster_calib(cluster_sum_E)

                preprocessed_data.append(cluster_calib_E)

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            file_num += self.num_procs
            print(f"Finished processing {file_num} files")

    def get_cluster_calib(self, cluster_E):
        """ Calibrate Clusters Energy """
        #works with cell or cluster array
            
        cluster_calib_E  = cluster_E / self.sampling_fraction

        if cluster_calib_E <= 0:
            return None

        return np.log10(cluster_calib_E)
