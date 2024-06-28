from multiprocessing        import Process, Value
from typing                 import Callable, Tuple, Any
from sklearn.neighbors      import NearestNeighbors

import numpy                as np
import compress_pickle      as pickle
import awkward              as ak
import uproot               as ur
import os
import logging
import tqdm
import gzip

from config_loader          import ConfigLoader
from exceptions             import DataPreprocessorException


class DataPreprocessor:

    """
    This class is responsible for preprocessing the data from the specified 
    directories in the config file. An instance of the LoadConfig class must
    be passed as an argument.
    """

    def __init__(
        self, 
        config_loader: ConfigLoader, 
        file_list: list, 
        folder_name: str,
        mask_function: Callable[[Any], Any]
    ):
        config = self.config = config_loader
        self.file_list = file_list
        self.folder_name = folder_name
        self.mask_function = mask_function
        self.num_files = len(file_list)
        
        means_file_path = self.config.OUTPUT_DIR_PATH / 'means.p.gz'
        stdvs_file_path = self.config.OUTPUT_DIR_PATH / 'stdvs.p.gz'

        if os.path.exists(means_file_path) and os.path.exists(stdvs_file_path):
            print("Using existing normalizer parameters")
            with gzip.open(config.OUTPUT_DIR_PATH / 'means.p.gz', 'rb') as means_file:
                self.means_dict = pickle.load(means_file)
            with gzip.open(config.OUTPUT_DIR_PATH / 'stdvs.p.gz', 'rb') as stdvs_file:
                self.stdvs_dict = pickle.load(stdvs_file)
        else:
            raise DataPreprocessorException("Normalizer stats do not exist")

        if config.PREPROCESS_DATA is True:
            print(f"Preprocessing {folder_name} folder data")
            self.data_preprocessor_manager()
        else:
            print(f"Using existing {folder_name} folder data")
            self.processed_file_list = [
                f"{config.OUTPUT_DIR_PATH}data_{file:03d}." 
                for file in range(self.num_files)
            ]


    """
    The below functions consist of the data preprocessor which assembles the
    graphs and the scalar preprocessor which computes the means and standard
    deviations of the dataset for z-score normalization of all of the event
    parameters.
    """

    
    def data_preprocessor_manager(self):

        config = self.config

        self.processes = []
        counter = Value('i', 0) 

        for i in range(config.NUM_PROCESSES):
            process = Process(
                target=self._data_preprocessor_process, 
                args=(i, counter), 
                daemon=True
            )
            process.start()
            self.processes.append(process)

        progress_bar = tqdm.tqdm(total=self.num_files, desc="Preprocessing Data")

        while any(process.is_alive() for process in self.processes):
            progress_bar.n = counter.value
            progress_bar.refresh()

        progress_bar.close()

        for process in self.processes:
            process.join()

        self.processed_file_list = [
            f"{config.OUTPUT_DIR_PATH}data_{file:03d}." 
            for file in range(self.num_files)
        ]


    def _data_preprocessor_process(self, worker_id: int, counter: Any):

        config = self.config
        file_num = worker_id

        while file_num < self.num_files:
            file_name = self.file_list[file_num]
            with ur.open(f"{file_name}:events") as events:
                event_data = events.arrays(["MCParticles.generatorStatus", "MCParticles.PDG",
                            'MCParticles.momentum.x', 'MCParticles.momentum.y', 'MCParticles.momentum.z',
                            config.DETECTOR_NAME+".energy", config.DETECTOR_NAME+".time",
                            config.DETECTOR_NAME+".position.x", config.DETECTOR_NAME+".position.y", config.DETECTOR_NAME+".position.z"])
                num_events = events.num_entries
            preprocessed_data = []
            
            for event_index in range(num_events):
                if config.OUTPUT_DIMENSIONS == 1:
                    target = self._get_momentum(event_data, event_index)
                elif config.OUTPUT_DIMENSIONS == 2:
                    target = self._get_momentum_theta(event_data, event_index)
                    if (target[1] * self.stdvs_dict["theta"] + self.means_dict["theta"]) > self.theta_max:
                        continue
                elif config.OUTPUT_DIMENSIONS == 3:
                    target = self._get_momentum_theta_phi(event_data, event_index)
                
                if config.USE_CLASSIFICATION:
                    particle_type = self._get_particle_type(event_data, event_index, file_name)
                    target += (particle_type,)

                nodes, global_node, cluster_num_nodes = self._get_graph_nodes(event_data, event_index)

                if cluster_num_nodes < 2:
                    senders, receivers, edges = None, None, None
                    continue
                else:
                    senders, receivers, edges = self._get_graph_edges(event_data, event_index, cluster_num_nodes)
                
                if not global_node:
                    continue

                graph = {
                    'nodes': nodes.astype(np.float64), 
                    'globals': global_node.astype(np.float64),
                    'senders': senders, 
                    'receivers': receivers, 
                    'edges': edges
                } 

                meta_data = [file_name]
                meta_data.extend(self._get_meta(event_data, event_index))
                preprocessed_data.append((graph, target, meta_data))

            data_dir_path = config.OUTPUT_DIR_PATH.resolve() / self.folder_name
            data_dir_path.mkdir(parents=True, exist_ok=True)

            with gzip.open(data_dir_path / f'data_{file_num:03d}.p.gz', 'wb') as data_file:
                pickle.dump(preprocessed_data, data_file)

            with counter.get_lock():
                counter.value += 1

            file_num += config.NUM_PROCESSES

    """
    The below functions are methods responsible for assembling 
    the graphs.
    """

    def _get_graph_nodes(self, event_data, event_index):
        nodes = self._get_cell_data(event_data[event_index])
        cluster_num_nodes = len(nodes)
        global_node = self._get_cluster_calibration_node(event_data[event_index])
        global_node = np.array([global_node])

        return nodes, global_node, cluster_num_nodes


    def _get_graph_edges(self, event_data, event_index, num_nodes):
        config = self.config
        
        cell_energy = ak.values_astype(event_data[event_index][config.DETECTOR_NAME + ".energy"], np.float64)
        time = ak.values_astype(event_data[event_index][config.DETECTOR_NAME + ".time"], np.float64)
        mask = (
            (config.ENERGY_TH < cell_energy) & 
            (time < config.TIME_TH) & 
            (cell_energy < 1e10)
        )

        if config.INCLUDE_ECAL is True:
            cell_energy_ecal = ak.values_astype(event_data[event_index][config.DETECTOR_ECAL + ".energy"], np.float64)
            time_ecal = ak.values_astype(event_data[event_index][config.DETECTOR_ECAL + ".time"], np.float64)
            mask_ecal = (
                (cell_energy_ecal > config.ENERGY_TH_ECAL) & 
                (time_ecal < config.TIME_TH) &
                (cell_energy_ecal < 1e10) 
            )

        node_features = []

        for feature in config.EDGE_FEATURE_NAMES:
            feature_data = ak.values_astype(event_data[event_index][config.DETECTOR_NAME + feature][mask], np.float64)
            feature_data = (feature_data - self.means_dict[feature])/self.stdvs_dict[feature]

            if config.INCLUDE_ECAL is True:
                feature_data_ecal = ak.values_astype(event_data[event_index][config.DETECTOR_ECAL + feature][mask_ecal], np.float64)
                feature_data_ecal = (feature_data - self.means_dict[feature])/self.stdvs_dict[feature]
                feature_data = np.concatenate((feature_data, feature_data_ecal))

            node_features.append(feature_data)
        
        node_features = np.swapaxes(node_features, 0, 1)

        if len(node_features) != num_nodes:
            raise DataPreprocessorException(
                f"Mismatch between number of nodes {len(node_features)}!={num_nodes}"
            )

        # Using k Nearest Neighbors on cell positions for creating graph
        curr_k = np.min([config.NUM_NEAREST_NEIGHBORS, num_nodes])

        neighbors = NearestNeighbors(n_neighbors=curr_k, algorithm='ball_tree')
        neighbors.fit(node_features)
        distances, indices = neighbors.kneighbors(node_features)
        
        senders = indices[:, 1:].flatten().astype(np.int64)
        receivers = np.repeat(indices[:, 0], curr_k - 1).astype(np.int64)
        edges = distances[:, 1:].reshape(-1, 1).astype(np.float64)

        return senders, receivers, edges


    def _get_cell_data(self, event_data):
        config = self.config

        cell_data = []
        cell_energy = ak.values_astype(event_data[config.DETECTOR_NAME + ".energy"], np.float64)
        time = ak.values_astype(event_data[config.DETECTOR_NAME + ".time"], np.float64)
        mask = (
            (cell_energy > config.ENERGY_TH) & 
            (time < config.TIME_TH) & 
            (cell_energy < 1e10)
        )

        if config.INCLUDE_ECAL is True:
            cell_data_ecal = []
            cell_energy_ecal = ak.values_astype(event_data[config.DETECTOR_ECAL + ".energy"], np.float64)
            time_ecal = ak.values_astype(event_data[config.DETECTOR_ECAL + ".time"], np.float64)
            mask_ecal = (
                (cell_energy_ecal > config.ENERGY_TH_ECAL) & 
                (time_ecal < config.TIME_TH) & 
                (cell_energy_ecal < 1e10) 
            )
            
        for feature in config.NODE_FEATURE_NAMES:
            feature_data = ak.values_astype(event_data[config.DETECTOR_NAME + feature][mask], np.float64)
            if "energy" in feature:  
                feature_data = np.log10(feature_data)
                feature_data = (feature_data - self.means_dict[config.DETECTOR_NAME + feature])/self.stdvs_dict[config.DETECTOR_NAME+feature]
            else:
                feature_data = (feature_data - self.means_dict[feature])/self.stdvs_dict[feature]

            cell_data.append(feature_data)

            if config.INCLUDE_ECAL is True:
                feature_data_ecal = ak.values_astype(event_data[config.DETECTOR_ECAL + feature][mask_ecal], np.float64)

                if "energy" in feature:
                    feature_data_ecal = np.log10(feature_data_ecal)
                    feature_data_ecal = (feature_data_ecal - self.means_dict[config.DETECTOR_ECAL + feature])/self.stdvs_dict[config.DETECTOR_ECAL+feature]
                else:
                    feature_data_ecal = (feature_data_ecal - self.means_dict[feature])/self.stdvs_dict[feature]

                cell_data_ecal.append(feature_data_ecal)

        cell_data = np.swapaxes(cell_data, 0, 1)

        if config.INCLUDE_ECAL is True:
            cell_data_ecal = np.swapaxes(cell_data_ecal, 0, 1)
            col_with_zero_ecal = np.zeros((cell_data_ecal.shape[0], 1))
            cell_data_ecal = np.hstack((cell_data_ecal, col_with_zero_ecal))

            col_with_one_hcal = np.ones((cell_data.shape[0], 1))

            cell_data = np.hstack((cell_data, col_with_one_hcal))
            cell_data = np.vstack((cell_data, cell_data_ecal))

        return cell_data


    def _get_cluster_calibration_node(self, event_data):

        config = self.config

        cell_energy = ak.values_astype(event_data[config.DETECTOR_NAME+".energy"], np.float64)
        cluster_calibration_energy = np.sum(cell_energy, axis = -1)
        cluster_calibration_energy /= config.SAMPLING_FRACTION
        
        if config.INCLUDE_ECAL is True:
            cell_energy_ecal = ak.values_astype(event_data[config.DETECTOR_ECAL+".energy"], np.float64)
            cluster_calibration_E_ecal = np.sum(cell_energy_ecal, axis = -1)
            cluster_calibration_energy += cluster_calibration_E_ecal

        if cluster_calibration_energy <= 0:
            return None

        cluster_calibration_energy  = np.log10(cluster_calibration_energy)
        cluster_calibration_energy -= self.means_dict["cluster_energy"]
        cluster_calibration_energy /= self.stdvs_dict["cluster_energy"]
        
        return cluster_calibration_energy



    """
    The below functions are accessors for the returning the target values and
    meta data for momentum, theta, or phi.
    """

    def _get_momentum(self, event_data, event_index) -> np.ndarray:
        mask = self.mask_function(event_data)

        momentum_x = np.float64(event_data['MCParticles.momentum.x'][mask][event_index, 0])
        momentum_y = np.float64(event_data['MCParticles.momentum.y'][mask][event_index, 0])
        momentum_z = np.float64(event_data['MCParticles.momentum.z'][mask][event_index, 0])

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]

        return momentum


    def _get_momentum_theta(self, event_data, event_index) -> Tuple[any, any]:
        mask = self.mask_function(event_data)

        momentum_x = np.float64(event_data['MCParticles.momentum.x'][mask][event_index, 0])
        momentum_y = np.float64(event_data['MCParticles.momentum.y'][mask][event_index, 0])
        momentum_z = np.float64(event_data['MCParticles.momentum.z'][mask][event_index, 0])

        momentum = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        theta = np.arccos(momentum_z/momentum)*1000

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]

        return momentum, theta


    def _get_momentum_theta_phi(self, event_data, event_index) -> Tuple[any, any, any]:
        mask = self.mask_function(event_data)
        momentum_x = np.float64(event_data['MCParticles.momentum.x'][mask][event_index, 0])
        momentum_y = np.float64(event_data['MCParticles.momentum.y'][mask][event_index, 0])
        momentum_z = np.float64(event_data['MCParticles.momentum.z'][mask][event_index, 0])

        momentum = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        theta = np.arccos(momentum_z/momentum)*1000
        phi = np.arctan2(momentum_y, momentum_x)

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        phi = (phi -self.means_dict["phi"]) / self.stdvs_dict["phi"]

        return momentum, theta, phi


    def _get_particle_type(self, event_data, event_index) -> int:
        incident_mask = event_data["MCParticles.generatorStatus"] == 1
        pdg_data = event_data["MCParticles.PDG"][incident_mask][event_index]

        particle_id = pdg_data[0]
        num_particles = len(pdg_data)
        is_single_particle = (num_particles == 1)
        is_double_photon = (
            num_particles == 2 and 
            pdg_data[0] == 22 and 
            pdg_data[1] == 22
        )

        if particle_id == 22 and is_single_particle:
            return 0
        elif particle_id == 111 and is_single_particle:
            return 1
        elif is_double_photon:
            return 1
        else:
            raise DataPreprocessorException(
                "Incident particle is not a photon or pi0. Classification " 
                "not supported"
            )


    def _get_meta(self, event_data, event_index) -> list:
        """ 
        Reading meta data
        Returns senders, receivers, and edges    
        """ 
        meta_data = [] 
        meta_data.append(event_index)

        return meta_data
