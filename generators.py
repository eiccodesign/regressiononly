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
    def __init__(self,
                 file_list: list,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs = 32,
                 preprocess = False,
                 output_dir = None,
                 take_log10 = False):
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
        self.sampling_fraction = 0.02 #0.0098 for Insert
        self.take_log10 = take_log10

        self.nodeFeatureNames = [".energy",".position.x",".position.y",".position.z",]
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.meta_features = ['file_name', 'event_ind']
        # self.edgeFeatureNames = self.cellGeo_data.keys()[9:]
        # self.num_edgeFeatures = len(self.edgeFeatureNames)

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
                
                #FIXME: may want to pass cluster_sum as only node at first.

                nodes, global_node, cluster_num_nodes = self..get_nodes(event_data, event_ind)
                senders, receivers, edges = self.get_edges(cluster_num_nodes)

                graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                    'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                    'edges': edges.astype(np.float32)}

                target = get_GenP(event_data)

                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target))

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(
                            self.output_dir + f'data_{file_num:03d}.p', 'wb'), 
                        compression='gzip')

            file_num += self.num_procs
            print(f"Finished processing {file_num} files")


    def get_nodes(self):

        nodes = get_cell_data(event_data)
        cluster_num_nodes = len(nodes)
        global_node = get_cluster_calib(event_data)

        return nodes, np.array([global_node]), cluster_num_nodes


    def get_cell_data(event_data):

        cell_data = []
        for nodeFeatureName in self.nodeFeatureNames:
            cell_data.append(event_data[self.detector_name+nodeFeatureName])

        return np.swapaxes(cell_data,0,1) #Events, Features
    #alternative: cell_data = np.reshape(cell_data, (len(self.nodeFeatureNames), -1)).T


    def get_cluster_calib(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        cluster_sum_E = np.sum(cell_E,axis=-1) #global node feature later
        cluster_calib_E  = cluster_E / self.sampling_fraction

        if cluster_calib_E <= 0:
            return None

        if (take_log10):
            return np.log10(cluster_calib_E)
        else:
            return(cluster_calib_E)

    def get_edges(self, num_nodes):
        return None,None,None

    def get_GenP(self,event_data):
        
        genPx = event_data['MCParticles.momentum.x'][:,2]
        genPy = event_data['MCParticles.momentum.y'][:,2]
        genPz = event_data['MCParticles.momentum.z'][:,2]

        genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        return genP

    def get_meta(self, event_data, event_ind):
        """ 
        Reading meta data
        Returns senders, receivers, and edges    
        """ 
        #For Now, only holds event id. Only one cluster per event, and no eta/phi
        meta_data = [] 
        meta_data.append(event_ind)
        
        return meta_data

    #FIXME: rm meta data
    def preprocessed_worker(self, worker_id, batch_queue):
        batch_graphs = []
        batch_targets = []
        batch_meta = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_meta.append(file_data[i][2])
                    
                if len(batch_graphs) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
                    
                    batch_queue.put((batch_graphs, batch_targets, batch_meta))
                    
                    batch_graphs = []
                    batch_targets = []
                    batch_meta = []

            file_num += self.num_procs
                    
        if len(batch_graphs) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
            
            batch_queue.put((batch_graphs, batch_targets, batch_meta))

    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing is required for combined classification/regression models.')
        
    def check_procs(self):
        for p in self.procs:
            if p.is_alive(): return True
        
        return False

    def kill_procs(self):
        for p in self.procs:
            p.kill()

        self.procs = []


    def generator(self):
        # for file in self.file_list:
        batch_queue = Queue(2 * self.num_procs)
            
        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)
        
        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue
            
            yield batch
        
        for p in self.procs:
            p.join()
            
if __name__ == '__main__':
    data_dir = '/usr/workspace/pierfied/preprocessed/data/'
    out_dir = '/usr/workspace/pierfied/preprocessed/preprocessed_data/'
    pion_files = np.sort(glob.glob(data_dir+'user*.root'))
    
    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                  cellGeo_file=data_dir+'cell_geo.root',
                                  batch_size=32,
                                  shuffle=False,
                                  num_procs=32,
                                  preprocess=True,
                                  output_dir=out_dir)

    gen = data_gen.generator()
    
    from tqdm.auto import tqdm
     
    for batch in tqdm(gen):
        pass
        
    exit()
