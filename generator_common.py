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

#MIP=0.0006 ## GeV
MIP_ECAL=0.13

#theta_max=4.0
#energy_TH=0.5*MIP
energy_TH_ECAL=0.5*MIP_ECAL
#NHITS_MIN=2

#Change these for your usecase!

# data_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/data/'
# out_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/preprocessed_data/'

data_dir = '/usr/workspace/hip/eic/log10_Uniform_03-23/ECCE_HCAL_Files/hcal_pi+_log10discrete_1GeV-150GeV_10deg-30deg_07-23-23/'
out_dir = '/usr/WS2/karande1/eic/gitrepos/regressiononly/preprocessed_data/train/'


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
                 output_dim: int = 2,
                 hadronic_detector: str = None,
                 include_ecal: bool = True,
                 k: int = 5):
        """Initialization"""

        self.preprocess = preprocess
        self.already_preprocessed = already_preprocessed
        self.calc_stats = calc_stats
        self.is_val = is_val
        self.output_dir = output_dir
        self.stats_dir = os.path.realpath(self.output_dir+'../')
        self.output_dim= output_dim

        os.makedirs(self.output_dir, exist_ok=True)

        self.hadronic_detector = hadronic_detector
        self.include_ecal = include_ecal

        self.file_list = file_list
        self.num_files = len(self.file_list)

        self.batch_size = batch_size
        self.num_features=num_features
        self.shuffle = shuffle
        
        self.num_procs = num_procs
        self.procs = []

        if(self.hadronic_detector=='hcal'):
            self.detector_name = "HcalEndcapPHitsReco"
            self.sampling_fraction =0.0224
            self.energy_TH=0.5*0.0006
            self.time_TH=150
            self.theta_max=1000.0
            
        elif(self.hadronic_detector=='insert'):    #'Insert' after the 'P'
            self.detector_name = "HcalEndcapPInsertHitsReco"
            self.sampling_fraction =0.0089
            self.energy_TH=0.5*0.0006
            self.time_TH=150
            self.theta_max=76.0

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
            
        

        self.detector_ecal='EcalEndcapPHitsReco'
        self.nodeFeatureNames = [".energy", ".position.z", ".position.x", ".position.y",]
        if self.output_dim==1:
            self.scalar_keys = [self.detector_name+self.nodeFeatureNames[0]] + \
                           self.nodeFeatureNames[1:] + \
                           ["clusterE","genP"] #, "theta"]
        elif self.output_dim==2:
            self.scalar_keys = [self.detector_name+self.nodeFeatureNames[0]] + \
                           self.nodeFeatureNames[1:] + \
                           ["clusterE","genP", "theta"]
        if self.include_ecal:
            self.scalar_keys = self.scalar_keys + [self.detector_ecal+self.nodeFeatureNames[0]]

        print('...............', self.num_features)
        # Slice the nodeFeatureNames list to only include the first 'num_features' elements
        self.nodeFeatureNames = self.nodeFeatureNames[:num_features]
        print(f'\n\n######################################')
        print(f'Using features: {self.nodeFeatureNames}') 
        print(f'######################################\n')
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_targetFeatures = 1 #Regression on Energy only for now

        self.edgeCreationFeatures = [".position.x", ".position.y", ".position.z", ]
        self.k = k
        self.num_edgeFeatures = 1   # edge length

        # if not self.is_val and self.calc_stats:
        if self.calc_stats:
            n_scalar_files = 8 #num files to use for scaler calculation
            self.preprocess_scalar(n_scalar_files)
        else:
            self.means_dict = pickle.load(open(f"{self.stats_dir}/means.p", 'rb'), compression='gzip')
            self.stdvs_dict = pickle.load(open(f"{self.stats_dir}/stdvs.p", 'rb'), compression='gzip')

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
        #event_data = event_tree.arrays(entry_stop=500) #need to use awkward
        print('____________XXXXXXXXXXXXXXXXXXXXXX', num_events)
        
        file_means = {k:[] for k in self.scalar_keys}
        file_stdvs = {k:[] for k in self.scalar_keys}
        
        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)

        if self.include_ecal:
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10) 

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
                    feature_data = np.log10(event_data[k][mask])

                file_means[k].append(np.mean(feature_data))
                file_stdvs[k].append(np.std(feature_data))
            else:
                continue

        cluster_sum_E_hcal = ak.sum(cell_E[mask],axis=-1) #global node feature later
        total_calib_E = cluster_sum_E_hcal / self.sampling_fraction
        #total_calib_E=total_calib_E[mask_theta]
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
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*1000  ## in milli radians

        
        if self.output_dim==1:
            file_means['genP'].append(ak.mean(genP))
            file_stdvs['genP'].append(ak.std(genP))
            
        if self.output_dim==2:
            mask_theta=theta<self.theta_max
            theta=theta[mask_theta]
            genP=genP[mask_theta]
            file_means['genP'].append(ak.mean(genP))
            file_stdvs['genP'].append(ak.std(genP))
            file_means['theta'].append(ak.mean(theta))
            file_stdvs['theta'].append(ak.std(theta))
        
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
            #for event_ind in range(6,20):    
                if self.output_dim==2:
                    target = self.get_GenP_Theta(event_data,event_ind)
                    if (target[1] * self.stdvs_dict["theta"] + self.means_dict["theta"])>self.theta_max:
                        continue
                elif self.output_dim==1:
                    target = self.get_GenP(event_data,event_ind)

                nodes, global_node, cluster_num_nodes = self.get_nodes(event_data, event_ind)
                if cluster_num_nodes<2:
                    senders, receivers, edges = None, None, None
                    continue
                else:
                    senders, receivers, edges = self.get_edges(event_data, event_ind, cluster_num_nodes)
                
                if not global_node:
                    continue

                graph = {'nodes': nodes.astype(np.float32), 
                         'globals': global_node.astype(np.float32),
                         'senders': senders, 
                         'receivers': receivers, 
                         'edges': edges} 

                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target, meta_data))

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            print(f"Finished processing file number {file_num}")
            file_num += self.num_procs


    def get_nodes(self, event_data, event_ind):

        nodes = self.get_cell_data(event_data[event_ind])
        cluster_num_nodes = len(nodes)
        global_node = self.get_cluster_calib(event_data[event_ind])

        return nodes, np.array([global_node]), cluster_num_nodes

    def get_cell_data(self,event_data):

        cell_data = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)

        if self.include_ecal:
            cell_data_ecal = []
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10) 

        for feature in self.nodeFeatureNames:

            feature_data = event_data[self.detector_name+feature][mask]
            if "energy" in feature:  
                feature_data = np.log10(feature_data)
                feature_data = (feature_data - self.means_dict[self.detector_name+feature])/self.stdvs_dict[self.detector_name+feature]
            else:
                feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            cell_data.append(feature_data)

            if self.include_ecal:
                feature_data_ecal = event_data[self.detector_ecal+feature][mask_ecal]
                if "energy" in feature:
                    feature_data_ecal = np.log10(feature_data_ecal)
                    feature_data_ecal = (feature_data_ecal - self.means_dict[self.detector_ecal+feature])/self.stdvs_dict[self.detector_ecal+feature]
                else:
                    feature_data_ecal = (feature_data_ecal- self.means_dict[feature]) / self.stdvs_dict[feature]
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
        cluster_calib_E = np.sum(cell_E,axis=-1)/self.sampling_fraction #global node feature later
        
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
        mask = (cell_E > self.energy_TH) & (time<self.time_TH) & (cell_E<1e10)

        if self.include_ecal:
            cell_E_ecal = event_data[event_ind][self.detector_ecal+".energy"]
            time_ecal = event_data[event_ind][self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<self.time_TH) & (cell_E_ecal<1e10) 

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

    
    def get_GenP_Theta(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*1000  #    *180/np.pi
        #gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
        #the generation has the parent praticle always at index 2

        genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        #gen_phi = (gen_phi - self.means_dict["phi"]) / self.stdvs_dict["phi"]
        return genP, theta

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
    pion_files = pion_files[2:10]
    # print("Pion Files = ",pion_files)

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=16,
                                    # calc_stats=True,
                                    preprocess=True,
                                    already_preprocessed=False,
                                    output_dir=out_dir,
                                    hadronic_detector="hcal",
                                    include_ecal=True,
                                    num_features=4)

    gen = data_gen.generator()

    print("\n~ DONE ~\n")
