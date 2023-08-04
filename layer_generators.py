#Imports.energy
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import uproot as ur
import awkward as ak
import time
from multiprocessing import Process, Queue, Manager, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
import random


#Change these for your usecase!
data_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/data/'
out_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/layering_preprocessed_data/'


class MPGraphDataGenerator:
    def __init__(self,
                 file_list: list,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs = 32,
                 n_Z_layers = 3,
                 calc_scalar= False,
                 preprocess = False,
                 nodeFeatureNames = None,
                 output_dir = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir

        self.file_list = file_list
        self.num_files = len(self.file_list)

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle: np.random.shuffle(self.file_list)

        self.num_procs = num_procs
        self.procs = []

        self.n_Z_layers = n_Z_layers
        self.z_min = 3800
        self.z_max = 5000.01
        self.z_range = self.z_max - self.z_min

        self.detector_name = "HcalEndcapPHitsReco" #'Insert' after the 'P'
        self.sampling_fraction = 0.02 #0.0098 for Insert

        self.nodeFeatureNames = nodeFeatureNames
        if (self.nodeFeatureNames is None):
            # self.nodeFeatureNames = [".energy"]
            self.nodeFeatureNames = ["layer_clusterE","layer.z",]

        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_targetFeatures = 1 #Regression on Energy only for now

        self.scalar_keys = self.nodeFeatureNames + ["genP"]

        # self.edgeFeatureNames = self.cellGeo_data.keys()[9:]
        # self.num_edgeFeatures = len(self.edgeFeatureNames)

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            n_scalar_files = 8 #num files to use for scaler calculation
            self.preprocess_scalar(n_scalar_files)
            self.preprocess_data()

        self.means_dict = pickle.load(open("./layering_preprocessed_data/means.p", 'rb'),compression='gzip')
        self.stdvs_dict = pickle.load(open("./layering_preprocessed_data/stdvs.p", 'rb'),compression='gzip')


    def preprocess_scalar(self,n_calcs):
        print(f'\nCalcing Scalars and saving data to {self.output_dir}')

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
            print("MEANS = ",means)
            print("STDVS = ",stdvs)

            print("SCALAR KEYS = " , self.scalar_keys)

            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))

            pickle.dump(self.means_dict, open(
                self.output_dir + 'means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                self.output_dir + 'stdvs.p', 'wb'), compression='gzip')

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


            file_means.append(0.5*(self.z_min+self.z_max))
            file_stdvs.append(self.z_range/np.sqrt(12)) #stdev of uniform distribution

            cell_E = event_data[self.detector_name+".energy"]
            cluster_sum_E = ak.sum(cell_E,axis=-1) #global node feature
            mask = cluster_sum_E > 0.0
            cluster_calib_E  = np.log10(cluster_sum_E[mask] / self.sampling_fraction)

            file_means.append(np.nanmean(cluster_calib_E)/self.n_Z_layers)
            file_stdvs.append(np.nanstd(cluster_calib_E)/self.n_Z_layers) #assumes uniform

            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            #generation has the parent particle at index 2

            file_means.append(ak.mean(genP))
            file_stdvs.append(ak.std(genP))

            means.append(file_means)
            stdvs.append(file_stdvs)

            file_num += self.num_procs


    def preprocess_data(self):
        print(f'\nPreprocessing and saving data to {self.output_dir}')

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
                # nodes, cluster_num_nodes = self.get_nodes(event_data, event_ind)
                senders, receivers, edges = self.get_edges(cluster_num_nodes) #returns 'None'

                graph = {
                    'nodes': nodes.astype(np.float32), 
                    # 'globals': global_node.astype(np.float32),
                    'senders': senders, 'receivers': receivers, 
                    'edges': edges
                } 

                target = self.get_GenP(event_data,event_ind)

                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target,meta_data))

            random.shuffle(preprocessed_data) 
            #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(
                self.output_dir + f'data_{file_num:03d}.p', 'wb'), 
                        compression='gzip')

            file_num += self.num_procs
            print(f"Finished processing {file_num} files")



    def get_nodes(self,event_data,event_ind):

        nodes = self.get_cell_data(event_data[event_ind]) #we pass in a single event
        cluster_num_nodes = len(nodes)
        # global_node = self.get_cluster_calib(event_data[event_ind])
        global_node = None 
        # print("NODES = ",nodes)

        return nodes, np.array([global_node]), cluster_num_nodes

    def get_Z_segmentation(self):

        random_layers = self.z_min + np.random.rand(self.n_Z_layers-1)*self.z_range

        return np.sort(random_layers)

    def get_segmented_cluster_sum(self, cell_e, z_indecies):

        segmented_cluster_sum = []
        for z_index in range(self.n_Z_layers):
            segmented_cluster_sum.append(np.sum(cell_e[z_indecies==z_index]))

        return np.asarray(segmented_cluster_sum)

    def get_cell_data(self,event_data):

        cell_data = []

        cell_E = event_data[self.detector_name+".energy"]
        mask = cell_E > 0.0


        z_layers = self.get_Z_segmentation()
        cell_z = event_data[self.detector_name+".position.z"]
        z_indecies = np.digitize(cell_z, z_layers) 
        #don't need to add edges (zmin, zmax) for digitize to work.

        segmented_cluster_sum = self.get_segmented_cluster_sum(cell_E[mask],z_indecies[mask])
        if np.sum(segmented_cluster_sum) == 0:
            return np.zeros(self.n_Z_layers + self.n_Z_layers-1) 
            #n_Z sums, n_Z-1 layer positions (no edges)

        # print("Z_Layers = ",z_layers)
        # print("Cluster Sums = ",segmented_cluster_sum)

        #for feature in self.nodeFeatureNames:

        #    feature_data = event_data[self.detector_name+feature][mask]

        #    if "energy" in feature:  
        #        feature_data = np.log10(feature_data)

        #    #standard scalar transform
        #    feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
        #    cell_data.append(feature_data)

        #cell_data.append(z_indecies)

        for sum in segmented_cluster_sum:
            cell_data.append(sum)
        for layer_posZ in z_layers:
            cell_data.append(layer_posZ)

        # print("Cell Data = ",cell_data)
        return np.asarray(cell_data)
        # return np.swapaxes(cell_data,0,1) # returns [cell_index, Features]
        #alternative: cell_data = np.reshape(cell_data, (len(self.nodeFeatureNames), -1)).T


    def get_cluster_calib(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        cluster_sum_E = np.sum(cell_E,axis=-1) #global node feature later
        cluster_calib_E  = cluster_sum_E / self.sampling_fraction

        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"]) / self.stdvs_dict["clusterE"]

        if cluster_calib_E <= 0:
            return None

        else:
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

            print("FILE DATA SHAPE = ",np.shape(file_data))

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
    pion_files = pion_files[:1]
    print("Pion Files = ",pion_files)

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=32,
                                    preprocess=True,
                                    output_dir=out_dir)

    gen = data_gen.generator()

    from tqdm.auto import tqdm

    for batch in tqdm(gen):
        pass

    print("\n~ DONE ~\n")
    exit()



def get_Z_masks(cell_z_array,z_layers):

    zmask = []
    for i in range(len(z_layers)-1):
        zmask.append(np.logical_and(cell_z_array >= z_layers[i], cell_z_array < z_layers[i+1]))

    return zmask
