#Imports
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import uproot as ur
import awkward as ak
import time
import sys
from multiprocessing import Process, Queue, Manager, set_start_method
# from multiprocess import Process, Queue, Manager, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, '/home/bishnu/bishnu/EIC/regressiononly/functions')
from Clusterer import *
import random
#MIP=0.0006 ## GeV
MIP_ECAL=0.13
epsilon=1e-10
rotation_angle=0.025
#time_TH=150  ## ns
#energy_TH=0.5*MIP
energy_TH_ECAL=0.5*MIP_ECAL
#NHITS_MIN=0
#z_min=3800      #3820
#z_max=5087.0      #50880 for ECCE
import uproot as ur
#Change these for your usecase!
# data_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/data/'
# out_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/preprocessed_data/'

#data_dir = '/usr/workspace/hip/eic/log10_Uniform_03-23/log10_pi+_Uniform_0-140Gev_17deg_1/'
#out_dir = '/usr/WS2/karande1/eic/gitrepos/regressiononly/preprocessed_data/'



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
                 #data_set: str =None,
                 output_dir: str = None,
                 num_features: int = 4,
                 output_dim: int =1,
                 hadronic_detector: str =None,
                 include_ecal: bool = True,
                 num_z_layers: int =10,
                 k: int = 5
    ):
        """Initialization"""

        self.preprocess = preprocess
        self.already_preprocessed = already_preprocessed
        self.calc_stats = calc_stats
        self.is_val = is_val
        #self.data_set=data_set
        self.hadronic_detector=hadronic_detector
        self.include_ecal=include_ecal
        self.output_dir = output_dir
        #self.stats_dir = os.path.realpath(self.output_dir)
        self.stats_dir = os.path.realpath(self.output_dir+'../')
        
        #self.val_stat_dir = os.path.dirname(self.stats_dir)
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.output_dim=output_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_z_layers=num_z_layers
        self.num_procs = num_procs
        self.procs = []
        print('n_Z layers in generators' , self.num_z_layers)
        print(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        if(self.hadronic_detector=='hcal'):
            self.detector_name = "HcalEndcapPHitsReco"
            self.sampling_fraction =0.0224
            self.time_TH=150  ## ns
            self.energy_TH=0.5*0.0006
            self.theta_max=1000.0
            
        elif(self.hadronic_detector=='hcal_insert'):    #'Insert' after the 'P'
            self.detector_name = "HcalEndcapPInsertHitsReco"
            self.sampling_fraction =0.0089
            self.time_TH=150  ## ns
            self.energy_TH=0.5*0.0006
            
        elif(self.hadronic_detector=='zdc_Fe'):  ##added by smoran
            self.detector_name = "ZDCHcalHitsReco"
            self.sampling_fraction =0.0203   ## CHANGE THIS NUMBER?
            self.energy_TH=0.5*0.000472
            self.time_TH=275
            self.theta_max=4.0

        elif(self.hadronic_detector=='zdc_Pb'):  ##added by smoran
            self.detector_name = "ZDCHcalHitsReco"
            self.sampling_fraction =0.0216   ## CHANGE THIS NUMBER?
            self.energy_TH=0.5*0.000393
            self.time_TH=275
            self.theta_max=4.0

            
        self.nodeFeatureNames = [".energy",".position.z", ".position.x",".position.y",]
        self.nodeFeatureNames_ecal =['ecal_energy','ecal_posz', 'ecal_posx', 'ecal_posy']
        self.detector_ecal='EcalEndcapPHitsReco'
        self.num_nodeFeatures = num_features
        

        # Slice the nodeFeatureNames list to only include the first 'num_features' elements
        ## SET UP FOR ONE/TWO DIMENSION OUTPUT AND WITH/WITHOUT ECAL
        self.nodeFeatureNames = self.nodeFeatureNames[:num_features]
        self.nodeFeatureNames_ecal = self.nodeFeatureNames_ecal[:num_features]
        self.edgeCreationFeatures = [".position.x",
                                     ".position.y",
                                     ".position.z", ]
        self.k = k
        self.num_edgeFeatures = 1   # edge length
        ## COMPUTE Z_MIN AND Z_MAX USING THE ONE FILE
        f_name=self.file_list[0]
        event_tree = ur.open(f_name)['events']
        num_events = event_tree.num_entries
        event_data = event_tree.arrays()
        position_z = event_data[self.detector_name+'.position.z']
        self.z_min=np.min(ak.flatten(position_z))
        self.z_max=np.max(ak.flatten(position_z))+0.5
        print(self.z_min, '[ zmin, zmax] ', self.z_max)


        
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_targetFeatures = output_dim   #Regression on Energy only (output dim =1)  Energy + theta for output_dim=2

        if ((self.num_targetFeatures==3) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP","theta", "phi"]

        elif ((self.num_targetFeatures==3) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+["clusterE","genP","theta", "phi"]
        
        elif ((self.num_targetFeatures==2) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP","theta"] #, "phi"]
            
        elif ((self.num_targetFeatures==2) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+["clusterE","genP","theta"] #,"phi"]
            
        elif ((self.num_targetFeatures==1) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP"]
            
        elif ((self.num_targetFeatures==1) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+ ["clusterE","genP"]    

        print('Sclar keys --------',self.scalar_keys)
        if self.calc_stats:
            n_scalar_files = 8 #num files to use for scaler calculation
            if include_ecal:
                self.preprocess_scalar_with_ecal(n_scalar_files)
            else:
                self.preprocess_scalar(n_scalar_files)
        else:
            self.means_dict = pickle.load(open(f"{self.stats_dir}/means.p", 'rb'), compression='gzip')
            self.stdvs_dict = pickle.load(open(f"{self.stats_dir}/stdvs.p", 'rb'), compression='gzip')
            print(self.means_dict)

        if self.already_preprocessed and os.path.isdir(self.output_dir):
            self.processed_file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]
            
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
            means = manager.list()
            stdvs = manager.list()
            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor, args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)

            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file dimension
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files
            #stdvs[stdvs == 0] = 1
            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            for key, value in self.stdvs_dict.items():
                if value==0:
                    self.stdvs_dict[key]=1
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
                p = Process(target=self.scalar_processor_with_ecal, args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)
                
            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file 
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files

            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            
            print(f"saving calc files to {self.stats_dir}/means.p\n")
            for key, value in self.stdvs_dict.items():
                
                if value==0:
                    self.stdvs_dict[key]=1
            
            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')
            
            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")

        
    def scalar_processor(self,worker_id,means,stdvs):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Mean + Stdev Calc. file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward
            #event_data = event_tree.arrays(entry_stop=500) #need to use awkward
            #print('xxxxxxxxxxxxxxxxxxx ',num_events)
            file_means = []
            file_stdvs = []
            cell_data=[]
            new_array=[]
            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10) 


            for feature_name in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature_name][mask]
                #if "energy" in feature_name:
                #    feature_data = np.log10(feature_data)
                max_length = max(len(sublist) for sublist in feature_data)
                padded_array = np.zeros((len(feature_data), max_length))
                for i, sublist in enumerate(feature_data):
                    padded_array[i, :len(sublist)] = sublist

                cell_data.append(padded_array)    
                    
            cell_data=np.array(cell_data)
            cell_data_swaped=np.swapaxes(cell_data,0,1)
            ## Arrange as E, Z, X, Y for all events
            for row in cell_data_swaped:
                column=np.column_stack((row[0], row[1], row[2], row[3]))
                new_array.append(column)
            new_array=np.array(new_array, dtype=object)
            
            ## Get Z segmentation regrouuping
            z_seg_array=[]
            cluster_sum_arr=[]
            for row in new_array:
                
                #if row.shape[0]<1:
                #    continue
                if np.all(row==0):
                    continue
                new_array=self.get_regrouped_zseg_unique_xy(self.num_z_layers, row)
                
                z_seg_array.append(new_array)
                
    
            z_seg_array=np.array(z_seg_array, dtype=object)
            z_seg_array=np.concatenate(z_seg_array)
            ## To avoid case of taking log10(of zero)
            z_seg_array[:, 0] = np.log10(z_seg_array[:, 0]+ epsilon)
            
            column_means=np.mean(z_seg_array, axis=0)
            column_stds=np.std(z_seg_array, axis=0)
            selected_column=[0, 1, 2, 3]
            for col in range(len(self.nodeFeatureNames)):
                file_means.append(column_means[col])
                file_stdvs.append(column_stds[col])
                           
            #unfortunatley, there's a version error so we can't use ak.nanmean...
            cluster_sum_E = ak.sum(cell_E[mask],axis=-1) #global node feature later
            
            #cluster_calib_E = ak.sum(cell_E[mask],axis=-1)            
            mask_sum = cluster_sum_E > 0.0
            #cluster_calib_E  =np.log10(cluster_sum_arr[mask]/self.sampling_fraction)
            #cluster_calib_E  =cluster_sum_E/self.sampling_fraction
            
            cluster_calib_E=np.log10(cluster_sum_E[mask_sum] / self.sampling_fraction)
            #print(self.sampling_fraction)
                        
            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))
            
            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            if not 'zdc' in self.hadronic_detector:
                genPx, genPz = self.rotateY(genPx, genPz, rotation_angle)  ## rotation w.r.t 25 mrad
            genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            #genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
            #generation has the parent particle at index 2

            if self.num_targetFeatures==1:
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))
            elif self.num_targetFeatures==2:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*1000
                mask_theta=theta<self.theta_max
                theta=theta[mask_theta]
                genP=genP[mask_theta]
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

            elif self.num_targetFeatures==3:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*1000
                gen_phi=(np.arctan2(genPy,genPx))*1000
                mask_theta=theta<self.theta_max
                theta=theta[mask_theta]
                genP=genP[mask_theta]
                gen_phi=gen_phi[mask_theta]
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))
                
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

                file_means.append(ak.mean(gen_phi))  ####
                file_stdvs.append(ak.std(gen_phi))   ####
            
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
            #last_entry = 100 #num_events
            #event_data = event_tree.arrays(entry_stop=500) #need to use awkward
            event_data = event_tree.arrays() #need to use awkward            
            
            file_means = []
            file_stdvs = []
            cell_data=[]
            new_array=[]
            
            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            
            mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)
            
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]

            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10) 
            
            
            for feature_name in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature_name][mask]
                                
                #if "energy" in feature_name:
                #    feature_data = np.log10(feature_data)
                max_length = max(len(sublist) for sublist in feature_data)
                padded_array = np.zeros((len(feature_data), max_length))
                for i, sublist in enumerate(feature_data):
                    padded_array[i, :len(sublist)] = sublist
                    
                cell_data.append(padded_array)
                
            cell_data=np.array(cell_data)
            cell_data_swaped=np.swapaxes(cell_data,0,1)
            for row in cell_data_swaped:
                column=np.column_stack((row[0], row[1], row[2], row[3]))
                new_array.append(column)
                
            new_array=np.array(new_array, dtype=object)
            ## Get Z segmentation regrouuping
            z_seg_array=[]
            cluster_sum_arr=[]
            for row in new_array:
                if np.all(row==0):
                    continue
                
                new_array=self.get_regrouped_zseg_unique_xy(self.num_z_layers, row)
                z_seg_array.append(new_array)
                
            z_seg_array=np.array(z_seg_array, dtype=object)
            z_seg_array=np.concatenate(z_seg_array)
            ##convert energy into log10
            z_seg_array[:, 0] = np.log10(z_seg_array[:, 0]+ epsilon) 
            
            
            column_means=np.mean(z_seg_array, axis=0)
            column_stds=np.std(z_seg_array, axis=0)
            selected_column=[0, 1, 2, 3]
            for col in range(len(self.nodeFeatureNames)):
                file_means.append(column_means[col])
                file_stdvs.append(column_stds[col])
            
                            
            ## ECAL MEANS AND STD AFTER HCAL     
            for feature_name in self.nodeFeatureNames:
                feature_data_ecal = event_data[self.detector_ecal+feature_name][mask_ecal]
                if "energy" in feature_name:
                    feature_data_ecal = np.log10(feature_data_ecal+epsilon)
                #if "position.z" in feature_name:
                #    #if feature_data_ecal!=3.72e+03:
                #    print(feature_data_ecal)
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
            #cluster_calib_E=np.log10(total_calib_E[mask])
            cluster_calib_E=np.log10(total_calib_E[mask])
            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))
            

            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            if not 'zdc' in self.hadronic_detector:
                genPx, genPz = self.rotateY(genPx, genPz, rotation_angle)  ## rotation w.r.t 25 mrad
            genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            #genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
            #generation has the parent particle at index 2
            if self.num_targetFeatures==1:
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))
            elif self.num_targetFeatures==2:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*1000
                mask_theta=theta<self.theta_max
                theta=theta[mask_theta]
                genP=genP[mask_theta]
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

            elif self.num_targetFeatures==3:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*1000
                gen_phi=(np.arctan2(genPy,genPx))*1000
                mask_theta=theta<self.theta_max
                theta=theta[mask_theta]
                genP=genP[mask_theta]
                gen_phi=gen_phi[mask_theta]
                file_means.append(np.mean(genP))
                file_stdvs.append(np.std(genP))

                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

                file_means.append(ak.mean(gen_phi))  ####
                file_stdvs.append(ak.std(gen_phi))   ####

            means.append(file_means)
            stdvs.append(file_stdvs)
            '''
            
            file_means.append(ak.mean(genP))
            file_stdvs.append(ak.std(genP))
            if self.num_targetFeatures==2:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*180/np.pi
                #gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

                #file_means.append(ak.mean(gen_phi))  ####
                #file_stdvs.append(ak.std(gen_phi))   ####
            means.append(file_means)
            stdvs.append(file_stdvs)
            '''
            file_num += self.num_procs
            

    
    def get_regrouped_zseg_unique_xy(self, num_z_layers, data):
        #z_layers = get_z_edges(n_Z_layers)
        z_layers=np.linspace(self.z_min,self.z_max,self.num_z_layers+1)
        z_centers = (z_layers[:-1] + z_layers[1:]) / 2
        #z_mask =  get_Z_masks(data[:,1],z_layers) #mask for binning
        
        new_array = []
        # Iterate over the bins of column 1
        for i in range(len(z_layers) - 1):
            # Filter the data for the current bin of column 1
            bin_data = data[(data[:, 1] >= z_layers[i]) & (data[:, 1] < z_layers[i + 1])]
            # Calculate the sum of column 1 for unique combinations of column 2 and column 3
            unique_sum = {}
            for row in bin_data:
                
                key = (row[2], row[3])
                if key not in unique_sum:
                    unique_sum[key] = 0
                unique_sum[key] += row[0]

            # Append the center value of the bin of column 1 and the sum values to the new array
            center_value = (z_layers[i] + z_layers[i + 1]) / 2
            for key, value in unique_sum.items():
                kera_array=np.column_stack((value, center_value, key[0], key[1]))
                new_array.append(kera_array)
        new_array=np.array(new_array)
        if new_array.shape[0]<1:
            new_array=np.zeros((data.shape[0], data.shape[1]))
            
        else:             
            new_array=np.swapaxes(new_array,1,2)
        new_array=np.reshape(new_array, (new_array.shape[0], new_array.shape[1]))
        return new_array

            
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
            #for event_ind in range(100,160):

                if self.num_targetFeatures==3:
                    target = self.get_GenP_Theta_Phi(event_data,event_ind)
                    if (target[1] * self.stdvs_dict["theta"] + self.means_dict["theta"])>self.theta_max:
                        continue

                elif self.num_targetFeatures==2:
                    target = self.get_GenP_Theta(event_data,event_ind)
                    if (target[1] * self.stdvs_dict["theta"] + self.means_dict["theta"])>self.theta_max:
                        continue
                    
                else:
                    target = self.get_GenP(event_data,event_ind)
                    
                nodess, global_node, cluster_num_nodes = self.get_nodes(event_data, event_ind)
                #print('Before regroupping   ')
                #print(nodess)
                if(self.include_ecal):

                    ## Get hcal data with last column , i.e '1' for hcal is removed from last column
                    node_only_hcal_temp=nodess[nodess[:,-1]==1]

                    ## Get ECAL data with last column i.e 0 is included
                    node_only_ecal=nodess[nodess[:,-1]==0]

                    ## GET hcal only without last column 1
                    node_only_hcal=node_only_hcal_temp[:,:-1]
                    nhits_hcal = np.sum(node_only_hcal_temp[:, -1] == 1)
                    node_only_hcal[:, 1:] = np.round(node_only_hcal[:, 1:], decimals=2)

                    node=self.get_regrouped_zseg_unique_xy(self.num_z_layers, node_only_hcal)
                    #print(node)
                    nodes_hcal=self.scalar_preprocessor_zseg(node)
                    
                    col_with_one_hcal=np.ones((nodes_hcal.shape[0],1))
                    cell_data_hcal_label=np.hstack((nodes_hcal, col_with_one_hcal))
                    nodes=np.vstack((cell_data_hcal_label, node_only_ecal))
                    
                    
                    nodes=nodes[:,[0,1,2,3,4]]
                                        
                ## only if HCAL  Only
                else:
                    nhits_hcal = nodess.shape[0]
                    nodess[:, 1:] = np.round(nodess[:, 1:], decimals=2)
                    node=self.get_regrouped_zseg_unique_xy(self.num_z_layers, nodess)
                    nodes=self.scalar_preprocessor_zseg(node)
                #print('event ', event_ind, '  Initial total ', nodess.shape[0], '   ecal  only  ', node_only_ecal.shape[0], ' hcal only Intial', node_only_hcal.shape[0],
                #      '  hcal_after regrouped  ', nodes_hcal.shape[0])
                #print('After regrouped ')
                #print(node)
                cluster_num_nodes_regrouped=nodes.shape[0]
                #print(event_ind, '  event and total hits after regroupping    ', cluster_num_nodes_regrouped)
                
                #senders, receivers, edges = self.get_edges(cluster_num_nodes) #returns 'None'
                if (cluster_num_nodes_regrouped<2):
                    
                    senders, receivers, edges = None, None, None
                    continue
                else:
                    senders, receivers, edges = self.get_edges(event_data, event_ind, cluster_num_nodes_regrouped)
                
                if None in global_node:
                    continue
                
                
                graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                    'senders': senders, 'receivers': receivers, 'edges': edges} 
                
                # graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                #     'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                #     'edges': edges.astype(np.float32)}
                
                    
                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target, meta_data)) 

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')
            
            print(f"Finished processing file number {file_num}")
            file_num += self.num_procs



    def get_nodes(self,event_data,event_ind):
        if(not self.include_ecal):
            nodes = self.get_cell_data(event_data[event_ind])
            global_node = self.get_cluster_calib(event_data[event_ind])
        if(self.include_ecal):
            nodes = self.get_cell_data_with_ecal(event_data[event_ind])
            global_node = self.get_cluster_calib_with_ecal(event_data[event_ind])
            
        cluster_num_nodes = len(nodes)
        return nodes, np.array([global_node]), cluster_num_nodes
    
    def get_cell_data(self,event_data):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)
        
        
        for feature in self.nodeFeatureNames:
            feature_data = event_data[self.detector_name+feature][mask]
            
            #if "energy" in feature:  
            #    feature_data = np.log10(feature_data)
                
            #standard scalar transform
            #feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            cell_data.append(feature_data)

        cell_data_swaped=np.swapaxes(cell_data,0,1)
        return cell_data_swaped
        #return np.swapaxes(cell_data,0,1) # returns [Events, Features]
        #alternative: cell_data = np.reshape(cell_data, (len(self.nodeFeatureNames), -1)).T
    def scalar_preprocessor_zseg(self, event_data):
        for index,feature in enumerate(self.nodeFeatureNames):
            if 'energy' in feature:
                event_data[:,index]= np.log10(event_data[:,index]+epsilon)
            
            #if self.stdvs_dict['.position.z']==0:  ## This is to adrress the case where the std is 0, for instance with z=1
            #    event_data[:,index]= (event_data[:,index] - self.means_dict[feature]) / 1.0
            #else:
            event_data[:,index]= (event_data[:,index] - self.means_dict[feature]) / self.stdvs_dict[feature]
            
        return event_data    

    ### WITH ECAL AND HCAL 
    def get_cell_data_with_ecal(self,event_data):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)
        

        cell_E_ecal = event_data[self.detector_ecal+".energy"]
        time_ecal=event_data[self.detector_ecal+".time"]
        mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10)
        #mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecalå<1e10)

        for feature in self.nodeFeatureNames:

            feature_data = event_data[self.detector_name+feature][mask]
            #feature_data_ecal = (feature_data_ecal - self.means_dict[feature_ecal]) / self.stdvs_dict[feature_ecal]
            
            #if "energy" in feature:
            #    feature_data = np.log10(feature_data)
            #    feature_data_ecal = np.log10(feature_data_ecal)
            #standard scalgetar transform
            #feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            #print('Mean hcal ll ', self.means_dict[feature])
            cell_data.append(feature_data)
            
        #print(cell_data_ecal)
        for feature, feature_ecal in zip(self.nodeFeatureNames, self.nodeFeatureNames_ecal):
            feature_data_ecal = event_data[self.detector_ecal+feature][mask_ecal]
            if "energy" in feature:
                    feature_data_ecal = np.log10(feature_data_ecal+epsilon)
            
            #if self.stdvs_dict[feature_ecal]==1:
            #    feature_data_ecal = (feature_data_ecal - self.means_dict[feature_ecal]) /1.0
            #else:    
            feature_data_ecal = (feature_data_ecal - self.means_dict[feature_ecal]) / self.stdvs_dict[feature_ecal]
                
            cell_data_ecal.append(feature_data_ecal)
        #print('after   ', cell_data_ecal)
        cell_data_swaped=np.swapaxes(cell_data,0,1)

        cell_data_ecal_swaped=np.swapaxes(cell_data_ecal,0,1)

        
        #cell_data_total=np.vstack((cell_data_swaped,cell_data_ecal_swaped)) 
        col_with_zero_ecal=np.zeros((cell_data_ecal_swaped.shape[0],1))
        cell_data_ecal_label=np.hstack((cell_data_ecal_swaped, col_with_zero_ecal))

        col_with_one_hcal=np.ones((cell_data_swaped.shape[0],1))
        cell_data_hcal_label=np.hstack((cell_data_swaped, col_with_one_hcal))

        cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        

        return cell_data_total

    def get_cluster_calib(self, event_data):
        nodes=self.get_cell_data(event_data)
        if np.all(nodes==0):
            pass
        else:
            nodes[:, 1:] = np.round(nodes[:, 1:], decimals=2)
            node=self.get_regrouped_zseg_unique_xy(self.num_z_layers, nodes)
            cluster_sum_E=np.sum(node[:,0])
        
            """ Calibrate Clusters Energy """
        
            #cell_E = event_data[self.detector_name+".energy"]
            #cluster_sum_E = np.sum(cell_E,axis=-1) #global node feature later
            if cluster_sum_E <= 0:
                return None
            #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        
            cluster_calib_E  = np.log10(cluster_sum_E/self.sampling_fraction)
            #cluster_calib_E  =cluster_sum_E/self.sampling_fraction
            cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]
        
            return(cluster_calib_E)
        


    ## WITH ECAL AND HCAL 
    def get_cluster_calib_with_ecal(self, event_data):
        """ Calibrate Clusters Energy """
                    
        cell_E = event_data[self.detector_name+".energy"]
        cell_E_ecal = event_data[self.detector_ecal+".energy"]
        
        cluster_sum_E_hcal = np.sum(cell_E,axis=-1) #global node feature later
        cluster_sum_E_ecal = np.sum(cell_E_ecal,axis=-1) #global node feature later

        cluster_calib_E_hcal  = cluster_sum_E_hcal/self.sampling_fraction
        cluster_calib_E_ecal  = cluster_sum_E_ecal
        
        #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        
        cluster_calib_E= cluster_calib_E_hcal + cluster_calib_E_ecal
        if cluster_calib_E<=0:
            return None
        cluster_calib_E=np.log10(cluster_calib_E)
        #cluster_calib_E=cluster_calib_E
        
        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]
        #print('Hello ', cluster_calib_E)
        return(cluster_calib_E)

    
    def get_regrouped_zseg_unique_xy_noE(self, num_z_layers, data, sum_E):
        z_layers=np.linspace(self.z_min,self.z_max,self.num_z_layers+1)
        z_centers = (z_layers[:-1] + z_layers[1:]) / 2
        #z_mask =  get_Z_masks(data[:,1],z_layers) #mask for binning
        new_array = []
	
        # Iterate over the bins of column 1
        #data=data[data[:,1].argsort()]
        for i in range(len(z_layers) - 1):
            # Filter the data for the current bin of column 1
            if sum_E:
                bin_data = data[(data[:, 1] >= z_layers[i]) & (data[:, 1] <= z_layers[i + 1])]
            else:
                bin_data = data[(data[:, 2] >= z_layers[i]) & (data[:, 2] < z_layers[i + 1])]

            unique_sum = {}
            for row in bin_data:
                if sum_E:
                    key = (row[2], row[3])
                    if key not in unique_sum:
                        unique_sum[key] = 0
                    unique_sum[key] += row[0]
                else:
                    key = (row[0], row[1])
                    if key not in unique_sum:
                        unique_sum[key] = 0
            # Append the center value of the bin of column 1 and the sum values to the new array
            center_value = (z_layers[i] + z_layers[i + 1]) / 2
            for key, value in unique_sum.items():
                if sum_E:
                    kera_array=np.column_stack((value, center_value, key[0], key[1]))
                else:
                    kera_array=np.column_stack(( key[0], key[1], center_value))
                new_array.append(kera_array)
        new_array=np.array(new_array)
        
        if new_array.shape[0]<1:
            new_array=np.zeros((data.shape[0], data.shape[1]))
            
        else:
            new_array=np.swapaxes(new_array,1,2)
        
        new_array=np.reshape(new_array, (new_array.shape[0], new_array.shape[1]))
        
        return new_array

    
    def get_edges(self, event_data, event_ind, num_nodes):
        cell_E = event_data[event_ind][self.detector_name+".energy"]
        time = event_data[event_ind][self.detector_name+".time"]
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)
        
        if self.include_ecal:
            cell_E_ecal = event_data[event_ind][self.detector_ecal+".energy"]
            time_ecal = event_data[event_ind][self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10)
            
        nodes_NN_feats_temp = []
        if self.include_ecal:
            nodes_NN_feats_temp_ecal = []
            for feature, ecal_mean in zip(self.edgeCreationFeatures, self.nodeFeatureNames_ecal):
                feature_data = event_data[event_ind][self.detector_name+feature][mask]
                #feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
                feature_data_ecal = event_data[event_ind][self.detector_ecal+feature][mask_ecal]
                if self.stdvs_dict[ecal_mean]==0:
                    feature_data_ecal = (feature_data_ecal - self.means_dict[ecal_mean]) / 1.0
                else:
                    feature_data_ecal = (feature_data_ecal - self.means_dict[ecal_mean]) / self.stdvs_dict[ecal_mean]
                    #feature_data = np.concatenate((feature_data, feature_data_ecal))

                nodes_NN_feats_temp.append(feature_data)
                nodes_NN_feats_temp_ecal.append(feature_data_ecal)
            nodes_NN_feats_temp_ecal = np.swapaxes(nodes_NN_feats_temp_ecal, 0, 1)  

        
        elif  self.include_ecal is False:
            for feature in self.edgeCreationFeatures:
                feature_data = event_data[event_ind][self.detector_name+feature][mask]
                #feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature
                nodes_NN_feats_temp.append(feature_data)
                #nodes_NN_feats_temp_ecal.append(feature_data_ecal)
            nodes_NN_feats_temp = np.swapaxes(nodes_NN_feats_temp, 0, 1)
        
        nodes_NN_feats_temp[:, 0:] = np.round(nodes_NN_feats_temp[:, 0:], decimals=2)
        nodes_NN_feats=self.get_regrouped_zseg_unique_xy_noE( self.num_z_layers, nodes_NN_feats_temp, False)
        if self.include_ecal:
            nodes_NN_feats=np.vstack((nodes_NN_feats, nodes_NN_feats_temp_ecal))
    
        for index, feature in enumerate(self.edgeCreationFeatures):
            if self.stdvs_dict[feature]==0:
                self.stdvs_dict[feature]=1
            nodes_NN_feats[:,index]=(nodes_NN_feats[:,index] - self.means_dict[feature]) / self.stdvs_dict[feature]

        #print('event  ' , event_ind, '  ' , len(nodes_NN_feats), '  hcal  ', nodes_NN_feats.shape[0], '  ecal ', nodes_NN_feats_temp_ecal.shape[0])    
        #assert len(nodes_NN_feats)==num_nodes, f"Mismatch between number of nodes {len(nodes_NN_feats)}!={num_nodes}"
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
        #genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        return genP
    def get_GenP_Theta(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        if not 'zdc' in self.hadronic_detector:
            genPx, genPz=self.rotateY(genPx, genPz, rotation_angle)
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*1000  #    *180/np.pi
        #gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        #gen_phi = (gen_phi - self.means_dict["phi"]) / self.stdvs_dict["phi"]
        return genP, theta
    
    def get_GenP_Theta_Phi(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        if not 'zdc' in self.hadronic_detector:
            genPx, genPz=self.rotateY(genPx, genPz, rotation_angle)
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*1000  #    *180/np.pi
        gen_phi=(np.arctan2(genPy,genPx))*1000    #mrad
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        gen_phi = (gen_phi - self.means_dict["phi"]) / self.stdvs_dict["phi"]
        return genP, theta, gen_phi

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
    
    ### Make everything with respective to z' (w.r.t) proton axis
    def rotateY(self, xdata, zdata, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        rotatedz = c*zdata - s*xdata
        rotatedx = s*zdata + c*xdata
        return rotatedx, rotatedz 

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
    # print("Pion Files = ",pion_files)

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=32,
                                    preprocess=True,
                                    already_preprocessed=True,
                                    output_dir=out_dir,
                                    num_features=num_features,
                                    output_dim=output_dim,
                                    hadronic_detector=hadronic_detector,
                                    include_ecal= True,
                                    num_z_layers=num_z_layers
    )

    gen = data_gen.generator()

    print("\n~ DONE ~\n")
    exit()