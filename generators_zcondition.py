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
import random

import sys
sys.path.insert(0, './functions')
from binning_utils import *

MIP=0.0006 ## GeV
MIP_ECAL=0.13
time_TH=150  ## ns
energy_TH=0.5*MIP
energy_TH_ECAL=0.5*MIP_ECAL
NHITS_MIN=0

data_dir = '/pscratch/sd/f/fernando/data_dir/'
out_dir = './'


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
                 data_set: str =None,
                 output_dir: str = None,
                 num_features: int = 4,
                 output_dim: int =1,
                 hadronic_detector: str =None,
                 include_ecal: bool = False,
                 condition_z = True,
                 num_z_layers = 5):
        """Initialization"""

        self.preprocess = preprocess
        self.already_preprocessed = already_preprocessed
        self.calc_stats = calc_stats
        self.is_val = is_val
        self.data_set=data_set
        self.hadronic_detector=hadronic_detector
        self.include_ecal=include_ecal
        self.output_dir = output_dir
        self.stats_dir = os.path.realpath(self.output_dir)
        self.val_stat_dir = os.path.dirname(self.stats_dir)
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.output_dim=output_dim
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.condition_z = condition_z
        self.num_z_layers = num_z_layers
        
        self.num_procs = num_procs
        self.procs = []


        self.detector_name = "HcalEndcapPHitsReco"
        self.sampling_fraction =0.0224

        if(self.hadronic_detector=='hcal'):
            self.detector_name = "HcalEndcapPHitsReco"
            self.sampling_fraction =0.0224
        elif(self.hadronic_detector=='hcal_insert'):    #'Insert' after the 'P'
            self.detector_name = "HcalEndcapPInsertHitsReco"
            self.sampling_fraction =0.0089
            
        
        self.nodeFeatureNames = [".energy",".position.z", 
                                 ".position.x",".position.y",]

        self.nodeFeatureNames_ecal =['ecal_energy','ecal_posz', 
                                     'ecal_posx', 'ecal_posy']


        #hcal z edges. Annoying to run here, but need z_edges early
        # self.edgesZ = self.get_original_Zedges(self.detector_name)
        self.edgesX, self.edgesY, self.edgesZ = self.get_original_edges('HcalEndcapPHitsReco')

        self.detector_ecal='EcalEndcapPHitsReco'
        self.num_nodeFeatures = num_features

        # Slice features for 1-4D cell info
        self.nodeFeatureNames = self.nodeFeatureNames[:num_features]
        self.nodeFeatureNames_ecal = self.nodeFeatureNames_ecal[:num_features]
        self.num_nodeFeatures = len(self.nodeFeatureNames)

        # Number of Predictions. E only if 1. Theta then Phi
        self.num_targetFeatures = output_dim 
        
        # Add conditional information to input features
        if ((self.num_targetFeatures==2) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP","theta"]
            
        elif ((self.num_targetFeatures==2) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+["clusterE","genP","theta"]
            
        elif ((self.num_targetFeatures==1) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP"]
            
        elif ((self.num_targetFeatures==1) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+ ["clusterE","genP"]    

        if ((self.num_targetFeatures==2) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP","theta"]

        #Add cluster_features to confige
        # cluster_features = ["clusterE", "genP", "theta"]  # loosely think of as global features
        # self.scalar_keys = self.nodeFeatureNames

        # if self.include_ecal:
        #     self.scalar_keys += self.self.nodeFeatureNames_ecal

        # for ifeat in range(num_targetFeatures): 
        #     self.scalar_keys += cluster_features(ifeat)
        print("\n SCALER KEYS = \n",self.scalar_keys) #for testing small refactor later
            

        # Use training mean and std for val and test
        if self.data_set!='val':
            if (self.calc_stats):
                n_scalar_files = 1 # num files to use for scaler calculation
                if(not self.include_ecal):
                    self.preprocess_scalar(n_scalar_files)  # 1st potential place

                elif (self.include_ecal):
                    self.preprocess_scalar_with_ecal(n_scalar_files)  # 1st potential place    
            else:
                self.means_dict = pickle.load(
                    open(f"{self.stats_dir}/means.p", 'rb'), compression='gzip')

                self.stdvs_dict = pickle.load(
                    open(f"{self.stats_dir}/stdvs.p", 'rb'), compression='gzip')
                
        elif self.data_set=='val':
            self.means_dict = pickle.load(
                open(f"{self.val_stat_dir}/train/means.p", 'rb'), compression='gzip')

            self.stdvs_dict = pickle.load(
                open(f"{self.val_stat_dir}/train/stdvs.p", 'rb'), compression='gzip')
            
        
        if self.already_preprocessed and os.path.isdir(self.output_dir):
            self.file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]

        elif self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()
        else:
            print('Check preprocessing config!!')

        if self.shuffle: np.random.shuffle(self.file_list)

    def get_original_edges(self, detector):

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

    def preprocess_scalar(self,n_calcs):
        print(f'\nCalcing Scalars and saving data to {self.stats_dir}')
        self.n_calcs = min(n_calcs,self.num_files)

        with Manager() as manager:
            means = manager.list()
            stdvs = manager.list()
            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor, 
                            args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)

            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file dimension
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files

            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
            print(f"saving calc files to {self.stats_dir}/means.p\n")

            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')

        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")
        
    def preprocess_scalar_with_ecal(self,n_calcs):
        print(f'\nCalcing Scalars and saving data to {self.stats_dir}')
        self.n_calcs = min(n_calcs,self.num_files)
        
        with Manager() as manager:
            means = manager.list()
            stdvs = manager.list()
            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor_with_ecal, 
                            args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)
                
            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file 
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files

            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
            print(f"saving calc files to {self.stats_dir}/means.p\n")
            
            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')

        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")

        
    def scalar_processor(self,worker_id,means,stdvs):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Mean + Stdev Calc. file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            file_means = []
            file_stdvs = []

            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10) 


            if not self.condition_z:
                for feature_name in self.nodeFeatureNames:
                    feature_data = event_data[self.detector_name+feature_name][mask]
                    
                    if "energy" in feature_name:
                        feature_data = np.log10(feature_data)
                        
                    file_means.append(ak.mean(feature_data))
                    file_stdvs.append(ak.std(feature_data))


            else:

                #Load Cell Data. Need all Dims for Histo
                cellX = ak.ravel(event_data[self.detector_name+'.position.x'][mask])
                cellY = ak.ravel(event_data[self.detector_name+'.position.y'][mask])
                cellZ = ak.ravel(event_data[self.detector_name+'.position.z'][mask])
                # cell_E = cell_E[mask]

                rand_Zs = get_random_z_pos(self.edgesZ,self.num_z_layers+1)

                new_features = get_newZbinned_cells(np.ravel(cell_E[mask]),
                                                    cellX, cellY, cellZ, 
                                                    self.edgesX, self.edgesY,
                                                    rand_Zs)


                for ifeat in range(len(self.nodeFeatureNames)):
                    file_means.append(ak.mean(new_features[ifeat]))
                    file_stdvs.append(ak.std(new_features[ifeat]))

            #unfortunatley, there's a version error so we can't use ak.nanmean...
            cluster_sum_E = ak.sum(cell_E[mask],axis=-1) #global node feature later

            mask = cluster_sum_E > 0.0
            cluster_calib_E  = np.log10(cluster_sum_E[mask] / self.sampling_fraction)

            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))

            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            #generation has the parent particle at index 2

            file_means.append(ak.mean(genP))
            file_stdvs.append(ak.std(genP))
            if self.num_targetFeatures==2:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*180/np.pi
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

            means.append(file_means)
            stdvs.append(file_stdvs)

            file_num += self.num_procs


    def scalar_processor_with_ecal(self,worker_id,means,stdvs):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Mean + Stdev Calc. file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            file_means = []
            file_stdvs = []

            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)

            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecal<1e10) 


            for feature_name in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature_name][mask]

                if "energy" in feature_name:
                    feature_data = np.log10(feature_data)


                file_means.append(ak.mean(feature_data))
                file_stdvs.append(ak.std(feature_data))

            ## ECAL MEANS AND STD AFTER HCAL     
            for feature_name in self.nodeFeatureNames:
                feature_data_ecal = event_data[self.detector_ecal+feature_name][mask_ecal]
                if "energy" in feature_name:
                    feature_data_ecal = np.log10(feature_data_ecal)
                ### ECAL    
                file_means.append(ak.mean(feature_data_ecal))
                file_stdvs.append(ak.std(feature_data_ecal))

                #unfortunatley, there's a version error so we can't use ak.nanmean...

            cluster_sum_E_hcal = ak.sum(cell_E[mask],axis=-1) #global node feature later
            cluster_sum_E_ecal=ak.sum(cell_E_ecal[mask_ecal],axis=-1)

            cluster_calib_E_hcal = cluster_sum_E_hcal / self.sampling_fraction
            cluster_calib_E_ecal  = cluster_sum_E_ecal ## sampling fractionn crrrection is already done

            total_calib_E= cluster_calib_E_hcal + cluster_calib_E_ecal
            mask = total_calib_E > 0.0
            cluster_calib_E=np.log10(total_calib_E[mask])

            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))


            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            #generation has the parent particle at index 2

            file_means.append(ak.mean(genP))
            file_stdvs.append(ak.std(genP))
            if self.num_targetFeatures==2:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*180/np.pi
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

            means.append(file_means)
            stdvs.append(file_stdvs)

            file_num += self.num_procs


    def preprocess_data(self):
        print(f'\nPreprocessing and saving data to {os.path.realpath(self.output_dir)}')

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
            print(f"Processing file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            preprocessed_data = []

            for event_ind in range(num_events):

                nodes, global_node, cluster_num_nodes = self.get_nodes(event_data, event_ind)
                senders, receivers, edges = self.get_edges(cluster_num_nodes) 
                #returns 'None'

                # print("\n\n nodes = ",nodes)
                if not global_node.any():
                    continue
                if None in global_node:
                    continue

                # print("global node = ", global_node)

                graph = {'nodes': nodes.astype(np.float32), 
                         'globals': global_node.astype(np.float32),
                         'senders': senders, 'receivers': receivers, 
                         'edges': edges} 

                # graph = {'nodes': nodes.astype(np.float32), 
                #          'globals': global_node.astype(np.float32),
                #          'senders': senders.astype(np.int32), 
                #          'receivers': receivers.astype(np.int32),
                #          'edges': edges.astype(np.float32)}

                if self.num_targetFeatures==2:
                    target = self.get_GenP_Theta(event_data,event_ind)    

                else:
                    target = self.get_GenP(event_data,event_ind)

                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target, meta_data)) 

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            print(f"Finished processing file number {file_num}")
            file_num += self.num_procs



    def get_nodes(self,event_data,event_ind):

        if(not self.include_ecal):

            global_node = np.array([self.get_cluster_calib(event_data[event_ind])])
            # global_node = np.nan_to_num(global_node)
            #FIXME: have cluster calib return array...

            if (self.condition_z):
                rand_Zs = get_random_z_pos(self.edgesZ, self.num_z_layers+1)
                nodes = self.get_cell_data(event_data[event_ind], rand_Zs)
                rand_Zs_norm = (rand_Zs - self.means_dict['.position.z']) / self.stdvs_dict['.position.z']
                global_node = np.append(global_node,rand_Zs_norm)

            else:
                nodes = self.get_cell_data(event_data[event_ind])

        if(self.include_ecal):
            nodes = self.get_cell_data_with_ecal(event_data[event_ind])
            global_node = np.array([self.get_cluster_calib_with_ecal(event_data[event_ind])])

        cluster_num_nodes = len(nodes)
        return nodes, global_node, cluster_num_nodes

    def get_cell_data(self,event_data,rand_Zs=None):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]

        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)


        # print("\n\n get_cell_data: nodes = ", self.nodeFeatureNames)
        if not self.condition_z:
            for feature in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature][mask]

                if "energy" in feature:  
                    feature_data = np.log10(feature_data)

                #standard scalar transform
                feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
                cell_data.append(feature_data)

        elif rand_Zs is not None:
            cellX = ak.ravel(event_data[self.detector_name+'.position.x'][mask])
            cellY = ak.ravel(event_data[self.detector_name+'.position.y'][mask])
            cellZ = ak.ravel(event_data[self.detector_name+'.position.z'][mask])
            # print("\n\nCell E = ",cell_E)


            new_features = get_newZbinned_cells(np.ravel(cell_E[mask]),
                                                cellX, cellY, cellZ, 
                                                self.edgesX, self.edgesY,
                                                rand_Zs)

            for ifeat in range(len(self.nodeFeatureNames)):
                feature = self.nodeFeatureNames[ifeat]
                feature_data = (new_features[ifeat] - self.means_dict[feature]) / self.stdvs_dict[feature]
                feature_data = np.nan_to_num(feature_data)
                cell_data.append(feature_data)


        cell_data_swaped=np.swapaxes(cell_data,0,1)
        # print("GetCellData :",cell_data_swaped)
        return cell_data_swaped
        #return np.swapaxes(cell_data,0,1) # returns [Events, Features]
        #alternative: cell_data = np.reshape(cell_data, (len(self.nodeFeatureNames), -1)).T


    ### WITH ECAL AND HCAL 
    def get_cell_data_with_ecal(self,event_data):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)


        cell_E_ecal = event_data[self.detector_ecal+".energy"]
        time_ecal=event_data[self.detector_ecal+".time"]
        mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecal<1e10)
        #mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecalå<1e10)

        for feature in self.nodeFeatureNames:

            feature_data = event_data[self.detector_name+feature][mask]
            feature_data_ecal = event_data[self.detector_ecal+feature][mask_ecal]
            if "energy" in feature:
                feature_data = np.log10(feature_data)
                feature_data_ecal = np.log10(feature_data_ecal)
            #standard scalar transform
            feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            #print('Mean hcal ll ', self.means_dict[feature])
            cell_data.append(feature_data)


        for feature_ecal in self.nodeFeatureNames_ecal:            
            feature_data_ecal = (feature_data_ecal - self.means_dict[feature_ecal]) / self.stdvs_dict[feature_ecal]
            #print('Mean ECA:::::: ll ', self.means_dict[feature_ecal])
            cell_data_ecal.append(feature_data_ecal)

        cell_data_swaped=np.swapaxes(cell_data,0,1)

        cell_data_ecal_swaped=np.swapaxes(cell_data_ecal,0,1)
        col_with_zero_ecal=np.zeros((cell_data_ecal_swaped.shape[0],1))
        cell_data_ecal_label=np.hstack((cell_data_ecal_swaped, col_with_zero_ecal))

        col_with_one_hcal=np.ones((cell_data_swaped.shape[0],1))
        cell_data_hcal_label=np.hstack((cell_data_swaped, col_with_one_hcal))

        cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))

        return cell_data_total

    def get_cluster_calib(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        time = event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10) 
        cluster_sum_E = np.sum(cell_E[mask],axis=-1) #global node feature later
        if cluster_sum_E <= 0:
            return None
        #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        cluster_calib_E  = np.log10(cluster_sum_E/self.sampling_fraction)
        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]

        return(cluster_calib_E)



    ## WITH ECAL AND HCAL 
    def get_cluster_calib_with_ecal(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        cell_E_ecal = event_data[self.detector_ecal+".energy"]

        cluster_sum_E_hcal = np.sum(cell_E,axis=-1) #global node feature later
        cluster_sum_E_ecal = np.sum(cell_E_ecal,axis=-1) #global node feature later
        '''
        if cluster_sum_E_hcal <= 0:
            return None    

        if cluster_sum_E_ecal<=0:
            return None
        '''
        cluster_calib_E_hcal  = cluster_sum_E_hcal/self.sampling_fraction
        cluster_calib_E_ecal  = cluster_sum_E_ecal

        #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))

        cluster_calib_E= cluster_calib_E_hcal + cluster_calib_E_ecal
        if cluster_calib_E<=0:
            return None
        cluster_calib_E=np.log10(cluster_calib_E)

        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]

        return(cluster_calib_E)


    def get_edges(self, num_nodes):
        return None,None,None

    def get_GenP(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        return genP

    def get_GenP_Theta(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*180/np.pi
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        return genP, theta

    #FIXME: DELETE THIS AND TARGET SCALARS
    def get_cell_scalars(self,event_data):

        means = []
        stdvs = []

        for feature in nodeFeatureNames:
            means.append(np.nanmean(event_data[feature]))
            stdvs.append(np.nanstd( event_data[feature]))

        return means, stdvs


    def get_target_scalars(self,target):

        return np.nanmean(target), np.nanstd(target)


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
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            #print("FILE DATA SHAPE = ",np.shape(file_data))

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_meta.append(file_data[i][2])
                #print('generator.py   shape of file    ',np.shape(file_data[i][1]))
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

    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing is required for regression models.')

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
    pion_files = pion_files[:20]
    print("Pion Files = ",pion_files)
    num_features = 4
    output_dim = 1
    hadronic_detector = 'hcal'

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=32,
                                    calc_stats=True,
                                    preprocess=True,
                                    already_preprocessed=False,
                                    output_dir=out_dir,
                                    num_features=num_features,
                                    output_dim=output_dim,
                                    hadronic_detector=hadronic_detector,
                                    include_ecal= False)

    gen = data_gen.generator()

    print("\n~ DONE ~\n")
    exit()
