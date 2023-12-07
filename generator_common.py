#Imports
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import uproot as ur
import awkward as ak
import time
from multiprocessing import Process, Queue, Manager, set_start_method
# from multiprocess import Process, Queue, Manager, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
from sklearn.neighbors import NearestNeighbors
import random

import sys
sys.path.insert(0, './functions')
from binning_utils import *

MIP=0.0006 ## GeV
MIP_ECAL=0.13
time_TH=150  ## ns
energy_TH=0.5*MIP
energy_TH_ECAL=0.5*MIP_ECAL
NHITS_MIN=2

#This is specified here for running this .py file by itself.
#train_model.py will get these paths from the config file in configs/

# data_dir = '/pscratch/sd/f/fernando/regressiononly/pi0_data/'
# out_dir = '/pscratch/sd/f/fernando/regression_common/regressiononly/preprocessed_pi0_1L/generator_test/'

data_dir = '/pscratch/sd/f/fernando/ECCE_data/'
out_dir = '/pscratch/sd/f/fernando/regression_common/regressiononly/preprocessed/generator_test/'

# data_dir = '/usr/workspace/hip/eic/log10_Uniform_03-23/ECCE_HCAL_Files/hcal_pi+_log10discrete_1GeV-150GeV_10deg-30deg_07-23-23/'
# out_dir = '/usr/WS2/karande1/eic/gitrepos/regressiononly/preprocessed_data/train/'


class MPGraphDataGenerator:
    def __init__(self,
                 file_list: list,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs: int = 32,
                 calc_stats: bool = False,
                 preprocess: bool = False,
                 already_preprocessed: bool = False,
                 is_val: bool = False,
                 output_dir: str = None,
                 num_features: int = 4,
                 output_dim: int = 1,
                 hadronic_detector: str = None,
                 include_ecal: bool = True,
                 k: int = 5,
                 n_zsections = None,
                 condition_zsections = False):
        """Initialization"""

        self.preprocess = preprocess
        self.already_preprocessed = already_preprocessed
        self.calc_stats = calc_stats
        self.is_val = is_val
        self.output_dir = output_dir
        self.stats_dir = os.path.realpath(self.output_dir+'../')
        print(f"\n\n STATS DIR = {self.stats_dir}\n\n")
        self.output_dim= output_dim

        os.makedirs(self.output_dir, exist_ok=True)

        self.hadronic_detector = hadronic_detector
        self.include_ecal = include_ecal

        self.file_list = file_list
        self.num_files = len(self.file_list)

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_procs = num_procs
        self.procs = []

        if(self.hadronic_detector=='hcal'):
            self.detector_name = "HcalEndcapPHitsReco"
            self.sampling_fraction =0.0224
        elif(self.hadronic_detector=='hcal_insert'):    #'Insert' after the 'P'
            self.detector_name = "HcalEndcapPInsertHitsReco"
            self.sampling_fraction =0.0089
        elif(self.hadronic_detector=='zdc'):  ##added by smoran
            self.detector_name = "ZDCHcalHitsReco"
            self.sampling_fraction =0.0224  ## CHANGE THIS NUMBER?    

        self.nodeFeatureNames = [".energy", ".position.z",
                                 ".position.x", ".position.y",]

        self.scalar_keys = [self.detector_name+self.nodeFeatureNames[0]] + \
                           self.nodeFeatureNames[1:] + ["clusterE","genP"]    

        self.detector_ecal='EcalEndcapPHitsReco'
        if self.include_ecal:
            self.scalar_keys = self.scalar_keys + \
                [self.detector_ecal+self.nodeFeatureNames[0]]


        # Slice the nodeFeatureNames list to only include the first 'num_features' elements
        self.nodeFeatureNames = self.nodeFeatureNames[:num_features]
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_targetFeatures = 1 #Regression on Energy only for now
        print('\n')
        print('#'*80,f'\nUsing features: {self.nodeFeatureNames}') 
        print('#'*80,'\n')

        self.edgeCreationFeatures = [".position.x",
                                     ".position.y",
                                     ".position.z", ]
        self.k = k
        self.num_edgeFeatures = 1   # edge length


        # HCal Z-Segmentation (training, not conditioning):
        self.custom_z = False
        self.n_zsections = n_zsections
        if (self.n_zsections is not None):
            self.custom_z = True
            if self.custom_z and self.include_ecal:
                sys.exit("ERROR: Custom Z and include ECal NOT supported")
            self.edgesX, self.edgesY, self.edgesZ \
            = self.get_cell_boundaries('HcalEndcapPHitsReco')
            self.z_layers = get_equidistant_layers(self.edgesZ,
                                                   self.n_zsections)
            self.z_centers = (self.z_layers[0:-1] + self.z_layers[1:])/2

            print(f'\nCell Boundaries = {self.edgesZ} [{len(self.edgesZ)}]')
            print(f'\nLongitudinal Layers = {self.z_layers} [{len(self.z_layers)}]')


        # n_zsections for conditioning
        self.condition_zsections = condition_zsections

        # if not self.is_val and self.calc_stats:
        if self.calc_stats:
            n_scalar_files = 8 #num files to use for scaler calculation
            self.preprocess_scalar(n_scalar_files)

        else:
            self.means_dict = pickle.load(open(f"{self.stats_dir}/means.p",
                                               'rb'), compression='gzip')
            self.stdvs_dict = pickle.load(open(f"{self.stats_dir}/stdvs.p",
                                               'rb'), compression='gzip')

        if self.already_preprocessed and os.path.isdir(self.output_dir):
            self.processed_file_list = [self.output_dir + f'data_{i:03d}.p'\
                for i in range(self.num_files)]

        elif self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

        else:
            print('Check preprocessing config!!')

        if self.shuffle: np.random.shuffle(self.processed_file_list)



    def preprocess_scalar(self,n_calcs):

        print(f'\nCalcing Scalars and saving data to {self.stats_dir}')

        self.n_calcs = min(n_calcs,self.num_files)

        with Manager() as manager:
            means = manager.list()  # dict({k:[] for k in self.scalar_keys})
            stdvs = manager.list()  # dict({k:[] for k in self.scalar_keys})

            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor,
                            args=(i, means, stdvs), daemon=True)
                p.start()
                self.procs.append(p)

            for p in self.procs:
                p.join()

            # means = np.mean(means,axis=0) #avg means along file dimension
            # stdvs = np.mean(stdvs,axis=0) #avg stdvs from files

            means_dict = dict({k:[] for k in self.scalar_keys})
            stdvs_dict = dict({k:[] for k in self.scalar_keys})
            for m, s in zip(means, stdvs):
                for k in self.scalar_keys:
                    means_dict[k].append(m[k])
                    stdvs_dict[k].append(s[k])
            
            self.means_dict = {k: np.mean(v) for k, v in means_dict.items()}
            self.stdvs_dict = {k: np.mean(v) for k, v in stdvs_dict.items()}

            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
            print(f"saving calc files to {self.stats_dir}/means.p\n")

            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')

        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")


    def scalar_processor(self, worker_id, means, stdvs):

        file_num = worker_id

        # while file_num < self.n_calcs:
        print(f"Mean + Stdev Calc. file number {file_num}")
        f_name = self.file_list[file_num]

        event_tree = ur.open(f_name)['events']
        num_events = event_tree.num_entries
        event_data = event_tree.arrays() #need to use awkward

        file_means = {k:[] for k in self.scalar_keys}
        file_stdvs = {k:[] for k in self.scalar_keys}

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)

        if self.custom_z:
            cell_Z = event_data[self.detector_name+'.position.z'][mask]
            binned_cell_E, binned_mask = Sum_EinZbins(cell_E[mask], cell_Z, self.z_layers)

        if self.include_ecal:
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & \
                (time_ecal<time_TH) & (cell_E_ecal<1e10) 

        print("SCALAR KEYS = ",self.scalar_keys)
        for k in self.scalar_keys:
            # print(k)
            if 'position' in k:

                feature_data = event_data[self.detector_name+k][mask]

                if self.include_ecal:
                    feature_data_ecal = event_data[self.detector_ecal+k][mask_ecal]
                    feature_data = ak.concatenate([feature_data, feature_data_ecal])

                file_means[k].append(np.mean(feature_data))
                file_stdvs[k].append(np.std(feature_data))
            
            elif 'energy' in k:
                if 'Ecal' in k:  
                    feature_data = np.log10(event_data[k][mask_ecal])
                else:
                    if self.custom_z:
                        feature_data = np.log10(binned_cell_E)
                    else:
                        feature_data = np.log10(event_data[k][mask])

                file_means[k].append(np.mean(feature_data))
                file_stdvs[k].append(np.std(feature_data))
            else:
                continue

        cluster_sum_E_hcal = ak.sum(cell_E[mask],axis=-1) #global node feature later
        total_calib_E = cluster_sum_E_hcal / self.sampling_fraction
        
        if self.include_ecal:
            cluster_sum_E_ecal = ak.sum(cell_E_ecal[mask_ecal],axis=-1)
            total_calib_E = total_calib_E + cluster_sum_E_ecal ## sampling fractionn crrrection is already done

        mask = total_calib_E > 0.0
        cluster_calib_E = np.log10(total_calib_E[mask])

        file_means['clusterE'].append(np.mean(cluster_calib_E))
        file_stdvs['clusterE'].append(np.std(cluster_calib_E))
        
        genPx = event_data['MCParticles.momentum.x'][:,2]
        genPy = event_data['MCParticles.momentum.y'][:,2]
        genPz = event_data['MCParticles.momentum.z'][:,2]
        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))

        file_means['genP'].append(ak.mean(genP))
        file_stdvs['genP'].append(ak.std(genP))

        means.append(file_means)
        stdvs.append(file_stdvs)
        # print(f'\nMeans: {means}')
        # print(f'Stds: {stdvs}')

    def preprocess_data(self):
        print(f'\nPreprocessing and saving data to {os.path.realpath(self.output_dir)}')

        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
        
        for p in self.procs:
            p.join()

        self.processed_file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]


    def preprocessor(self, worker_id):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Processing file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            preprocessed_data = []

            for event_ind in range(num_events):

                nodes, global_node, cluster_num_nodes = self.get_nodes(event_data, event_ind)


                if cluster_num_nodes<2 or self.custom_z:
                    # senders, receivers, edges = None, None, None
                    senders, receivers, edges = None, None, None
                    # continue
                else:
                    senders, receivers, edges = self.get_edges(event_data, event_ind, cluster_num_nodes)
                
                # if not global_node:
                if None in global_node:
                    continue


                graph = {'nodes': nodes.astype(np.float32), 
                         'globals': global_node.astype(np.float32),
                         'senders': senders, 
                         'receivers': receivers, 
                         'edges': edges} 

                target = self.get_GenP(event_data,event_ind)

                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target, meta_data))

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            print(f"Finished processing file number {file_num}")
            file_num += self.num_procs


    def get_nodes(self, event_data, event_ind):

        global_node = self.get_cluster_calib(event_data[event_ind])

        if (self.condition_zsections):
            rand_Zs = get_random_z_pos(self.edgesZ, self.n_zsections+1)
            nodes = self.get_cell_data(event_data[event_ind], rand_Zs)
            rand_Zs_norm = (rand_Zs - self.means_dict['.position.z']) \
                / self.stdvs_dict['.position.z']
            global_node = np.append(global_node, rand_Zs_norm)

        else:
            nodes = self.get_cell_data(event_data[event_ind])

        # nodes = self.get_cell_data(event_data[event_ind])
        cluster_num_nodes = len(nodes)

        # return nodes, np.array([global_node]), cluster_num_nodes

        if not self.condition_zsections:
            global_node = np.array([global_node])

        return nodes, global_node, cluster_num_nodes


    def get_cell_data(self,event_data, n_zsections=None):

        cell_data = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)

        cell_E = cell_E[mask]

        if self.custom_z and n_zsections is not None:
            cell_Z = event_data[self.detector_name+'.position.z'][mask]
            binned_cell_E, binned_mask = Sum_EinZbins(cell_E, cell_Z, self.z_layers)
            binned_cell_Z = self.z_centers[binned_mask]

            cellX = ak.ravel(event_data[self.detector_name+'.position.x'][mask])
            cellY = ak.ravel(event_data[self.detector_name+'.position.y'][mask])
            cellZ = ak.ravel(event_data[self.detector_name+'.position.z'][mask])


            new_features = get_newZbinned_cells(np.ravel(cell_E),
                                                cellZ, cellX, cellY, 
                                                self.edgesX, self.edgesY,
                                                n_zsections)

            # print("%"*30)
            # print("New Z = ",new_features[1])
            for i_feat, feature in enumerate(self.nodeFeatureNames):
                if "energy" in feature: feature = self.detector_name + feature
                feature_data = (new_features[i_feat] - self.means_dict[feature])\
                    / self.stdvs_dict[feature]
                feature_data = np.nan_to_num(feature_data)
                cell_data.append(feature_data)
            cell_data = np.swapaxes(cell_data, 0, 1)
            return cell_data

        # ECAL and Z Conditioning not compatible at this time 10/3/23
        if self.include_ecal:
            cell_data_ecal = []
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & \
                (time_ecal<time_TH) & (cell_E_ecal<1e10) 

        # if self.custom_z:
        #     FIXME: START HERE TOMORROW
        #     get_new_cell stuff(z_layers)
        #     return new cell_stuff

        for feature in self.nodeFeatureNames:

            feature_data = event_data[self.detector_name+feature][mask]
            if self.custom_z:
                feature_data = binned_cell_Z
                #only works for cellZ for now. Do not pass XY

            if "energy" in feature:  
                if self.custom_z:
                    feature_data = binned_cell_E
                    #feature_data = Sum_EinZbins(feature_data, cellZ, self.z_layers)
                feature_data = np.log10(feature_data)
                hcal_feature = self.detector_name+feature
                #The energy feature specifies hcal or ecal in it's name

            #Standard Scalor
            feature_data = (feature_data - self.means_dict[hcal_feature])\
                / self.stdvs_dict[hcal_feature]


            cell_data.append(feature_data)

            if self.include_ecal:
                feature_data_ecal=event_data[self.detector_ecal+feature][mask_ecal]

                if "energy" in feature:
                    feature_data_ecal = np.log10(feature_data_ecal)
                    ecal_feature = self.detector_ecal+feature

                #standard scalar
                feature_data_ecal = (feature_data_ecal - \
                    self.means_dict[ecal_feature])\
                    / self.stdvs_dict[ecal_feature]

                cell_data_ecal.append(feature_data_ecal)


        cell_data = np.swapaxes(cell_data, 0, 1)
        
        if self.include_ecal:
            cell_data_ecal = np.swapaxes(cell_data_ecal, 0, 1)
            col_with_zero_ecal = np.zeros((cell_data_ecal.shape[0],1))
            cell_data_ecal = np.hstack((cell_data_ecal, col_with_zero_ecal))

            col_with_one_hcal = np.ones((cell_data.shape[0],1))
            cell_data = np.hstack((cell_data, col_with_one_hcal))

            cell_data = np.vstack((cell_data, cell_data_ecal))

        return cell_data # returns [Events, Features]


    def get_cluster_calib(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        cluster_calib_E = np.sum(cell_E,axis=-1)/self.sampling_fraction 
        #global node feature later
        
        if self.include_ecal:
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            cluster_calib_E_ecal = np.sum(cell_E_ecal,axis=-1)
            cluster_calib_E += cluster_calib_E_ecal

        if cluster_calib_E <= 0:
            return None

        cluster_calib_E  = np.log10(cluster_calib_E)
        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]
        return cluster_calib_E

    def get_edges(self, event_data, event_ind, num_nodes):
        
        cell_E = event_data[event_ind][self.detector_name+".energy"]
        time = event_data[event_ind][self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)


        if self.include_ecal:
            cell_E_ecal = event_data[event_ind][self.detector_ecal+".energy"]
            time_ecal = event_data[event_ind][self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecal<1e10) 

        nodes_NN_feats = []
        for feature in self.edgeCreationFeatures:
            feature_data = event_data[event_ind][self.detector_name+feature][mask]

            feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]

            if self.include_ecal:
                feature_data_ecal = event_data[event_ind][self.detector_ecal+feature][mask_ecal]
                feature_data_ecal = (feature_data_ecal - self.means_dict[feature]) / self.stdvs_dict[feature]

                feature_data = np.concatenate((feature_data, feature_data_ecal))


            nodes_NN_feats.append(feature_data)
        
        nodes_NN_feats = np.swapaxes(nodes_NN_feats, 0, 1)
        assert len(nodes_NN_feats)==num_nodes, f"Mismatch between number of nodes {len(nodes_NN_feats)}!={num_nodes}"
        # Using kNN on x, y, z for creating graph
        curr_k = np.min([self.k, num_nodes])

        # nbrs = NearestNeighbors(n_neighbors=curr_k, algorithm='ball_tree').fit(nodes[:, 2:5])
        nbrs = NearestNeighbors(n_neighbors=curr_k, algorithm='ball_tree').fit(nodes_NN_feats)
        distances, indices = nbrs.kneighbors(nodes_NN_feats)
        
        senders = indices[:, 1:].flatten()
        receivers = np.repeat(indices[:, 0], curr_k-1)
        edges = distances[:, 1:].reshape(-1, 1)
        return senders.astype(np.int32), receivers.astype(np.int32), edges.astype(np.float32)

    def get_GenP(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]

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

    def preprocessed_worker(self, worker_id, batch_queue):
        batch_graphs = []
        batch_targets = []
        batch_meta = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.processed_file_list[file_num], 'rb'), compression='gzip')

            # print("FILE DATA SHAPE = ",np.shape(file_data))

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_meta.append(file_data[i][2])

                # batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
                '''need the above line if there are more than 1 cluster per event'''

                if len(batch_graphs) == self.batch_size:

                    batch_queue.put((batch_graphs, batch_targets, batch_meta))

                    batch_graphs = []
                    batch_targets = []
                    batch_meta = []

            file_num += self.num_procs

        if len(batch_graphs) > 0:
            # batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)

            batch_queue.put((batch_graphs, batch_targets, batch_meta))

    def get_cell_boundaries(self, detector):

        #IMPORTANT: This won't work if a single root file
        #Does not contain at least one shower that reaches the back of
        #the calorimeter. pi0 and photon guns beware!

        event_tree = ur.open(self.file_list[0])['events']
        num_events = event_tree.num_entries
        event_data = event_tree.arrays() #need to use awkward

        cell_E = event_data[detector+'.energy']
        time = event_data[detector+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10) 

        cellX = ak.ravel(event_data[detector+'.position.x'][mask])
        cellY = ak.ravel(event_data[detector+'.position.y'][mask])
        cellZ = ak.ravel(event_data[detector+'.position.z'][mask])

        centersX, edgesX, widthX = get_bin_edges(cellX)
        centersY, edgesY, widthY = get_bin_edges(cellY)
        centersZ, edgesZ, widthZ = get_bin_edges(cellZ)

        return edgesX, edgesY, edgesZ


    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing required for regression models.')

    def check_procs(self):
        for p in self.procs:
            if p.is_alive(): return True

        return False

    def kill_procs(self):
        for p in self.procs:
            p.kill()

        self.procs = []


    def generator(self):
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

            #FIXME: Print Batches here too
            yield batch

        for p in self.procs:
            p.join()

if __name__ == '__main__':
    pion_files = np.sort(glob.glob(data_dir+'*.root')) #dirs L14
    pion_files = pion_files[2:10]

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=16,
                                    calc_stats=True,
                                    preprocess=True,
                                    already_preprocessed=False,
                                    output_dir=out_dir,
                                    hadronic_detector="hcal",
                                    include_ecal=False,
                                    num_features=2,
                                    n_zsections=8,
                                    condition_zsections = True)

    gen = data_gen.generator()

    print("\n~ DONE ~\n")
