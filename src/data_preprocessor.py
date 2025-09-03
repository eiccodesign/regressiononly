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

        detector_names = config.detector_names
        detector_dictionary = config.detector_dictionary

        while file_num < self.num_files:
            file_name, particle_name = self.file_list[file_num]
            if config.USE_CLASSIFICATION:
                particle_type = self._get_particle_type(particle_name)
            with ur.open(f"{file_name}:events") as events:
                branch_names = ["MCParticles.generatorStatus", "MCParticles.PDG",
                        'MCParticles.momentum.x', 'MCParticles.momentum.y', 'MCParticles.momentum.z', "MCParticles.mass"]
                for detector_name in detector_names:
                    detector_branch = detector_dictionary[detector_name]["BRANCH_NAME"]
                    branch_names += [
                        detector_branch+".energy",
                        detector_branch+".time",
                        detector_branch+".position.x",
                        detector_branch+".position.y",
                        detector_branch+".position.z"
                    ]
                event_data = events.arrays(branch_names)
                num_events = events.num_entries
            preprocessed_data = []
            

            for event_index in range(num_events):
                target = self._get_targets(event_data, event_index, particle_name)
                # Removing events that didn't pass the cuts or otherwise are empty
                if len(target) == 0:
                    continue
                if config.USE_CLASSIFICATION:
                    target += (particle_type,)
                # For 1D output, should just have a number for the target
                if len(target) == 1:
                    target = target[0]
                nodes, global_node, cluster_num_nodes = self._get_graph_nodes(event_data, event_index)

                if cluster_num_nodes < 2:
                    senders, receivers, edges = None, None, None
                    continue
                else:
                    senders, receivers, edges = self._get_graph_edges(event_data, event_index, cluster_num_nodes)
                
                if not global_node:
                    continue

                graph = {
                    'nodes': nodes.astype(np.float32), 
                    'globals': global_node.astype(np.float32),
                    'senders': senders, 
                    'receivers': receivers, 
                    'edges': edges
                } 

                meta_data = [file_name]
                meta_data.extend(self._get_meta(event_data, event_index))
                meta_data.extend([particle_name])
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
        
        detector_masks = {}

        node_features = []
        for detector_name in config.detector_names:
            branch_name = config.detector_dictionary[detector_name]["BRANCH_NAME"]
            cell_energy = ak.values_astype(event_data[branch_name+ ".energy"], np.float64)[event_index]
            time = event_data[branch_name + ".time"][event_index]
            mask = cell_energy < 1e10
            if config.detector_dictionary[detector_name]["ENERGY_TH"] is not None:
                energy_mask = cell_energy > config.detector_dictionary[detector_name]["ENERGY_TH"]
                mask = (mask) & (energy_mask)
            if config.detector_dictionary[detector_name]["TIME_TH"] is not None:
                time_mask = time < config.detector_dictionary[detector_name]["TIME_TH"]
                mask = (mask) & (time_mask)
            
            detector_masks[detector_name] = mask

        

        for feature in config.EDGE_FEATURE_NAMES:
            feature_data = np.array([])
            for detector_name in config.detector_names:
                branch_name = config.detector_dictionary[detector_name]["BRANCH_NAME"] + feature
                mask = detector_masks[detector_name]

                detector_feature_data = event_data[event_index][branch_name][mask]
                detector_feature_data = (detector_feature_data - self.means_dict[feature])/self.stdvs_dict[feature]
                feature_data = np.concatenate((feature_data, detector_feature_data))
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
        
        senders = indices[:, 1:].flatten().astype(np.int32)
        receivers = np.repeat(indices[:, 0], curr_k - 1).astype(np.int32)
        edges = distances[:, 1:].reshape(-1, 1).astype(np.float32)

        return senders, receivers, edges


    def _get_cell_data(self, event_data):
        config = self.config

        cell_data = []
        detector_masks = {}
        for detector_name in config.detector_names:
            branch_name = config.detector_dictionary[detector_name]["BRANCH_NAME"]
            cell_energy = ak.values_astype(event_data[branch_name+ ".energy"], np.float64)
            time = event_data[branch_name + ".time"]
            mask = cell_energy < 1e10
            if config.detector_dictionary[detector_name]["ENERGY_TH"] is not None:
                energy_mask = cell_energy > config.detector_dictionary[detector_name]["ENERGY_TH"]
                mask = (mask) & (energy_mask)
            if config.detector_dictionary[detector_name]["TIME_TH"] is not None:
                time_mask = time < config.detector_dictionary[detector_name]["TIME_TH"]
                mask = (mask) & (time_mask)
            
            detector_masks[detector_name] = mask
        
        ecal_simulated = False
        hcal_simulated = False
        for detector_name in config.detector_names:
            detector_type = config.detector_dictionary[detector_name]["DETECTOR_TYPE"]
            if detector_type == "HCAL":
                hcal_simulated = True
            elif detector_type == "ECAL":
                ecal_simulated = True
            else:
                raise ValueError(f"{detector_name} has unsupported detector type in config_loader.py!")
        
        for detector_name in config.detector_names:
            branch_name = config.detector_dictionary[detector_name]["BRANCH_NAME"]
            detector_cell_data = []
            for feature in config.NODE_FEATURE_NAMES:
                mask = detector_masks[detector_name]
                feature_data = event_data[branch_name + feature][mask]
                if "energy" in feature:  
                    feature_data = np.log10(feature_data)
                    feature_data = (feature_data - self.means_dict[branch_name + feature])/self.stdvs_dict[branch_name+feature]
                else:
                    feature_data = (feature_data - self.means_dict[feature])/self.stdvs_dict[feature]
                detector_cell_data.append(feature_data)
            
            detector_cell_data = np.swapaxes(detector_cell_data, 0, 1)
            if ecal_simulated and hcal_simulated:
                detector_type = config.detector_dictionary[detector_name]["DETECTOR_TYPE"]
                if detector_type == "ECAL":
                    detector_index = np.zeros((detector_cell_data.shape[0], 1))
                elif detector_type == "HCAL":
                    detector_index = np.ones((detector_cell_data.shape[0], 1))
                else:
                    raise ValueError(f"{detector_name} has unsupported detector type in config_loader.py!")

                detector_cell_data = np.hstack((detector_cell_data, detector_index))
            cell_data.append(detector_cell_data)
        cell_data = np.vstack(cell_data)
        return cell_data


    def _get_cluster_calibration_node(self, event_data):

        config = self.config

        cluster_calibration_energy = 0
        for detector_name in config.detector_names:
            branch_name = config.detector_dictionary[detector_name]["BRANCH_NAME"]
            cell_energy = event_data[branch_name+".energy"]
            detector_energy = np.sum(cell_energy, axis = -1)
            detector_energy /= config.detector_dictionary[detector_name]["SAMPLING_FRACTION"]
            cluster_calibration_energy += detector_energy

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
    def _rotateY(self, xdata, zdata, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        rotatedz = c*zdata - s*xdata
        rotatedx = s*zdata + c*xdata
        return rotatedx, rotatedz
    
    def _get_targets(self, event_data, event_index, particle_name) -> Tuple[Any, ...]:
        mask = self.mask_function(event_data, particle_name)

        momentum_x = event_data['MCParticles.momentum.x'][mask][event_index]
        momentum_y = event_data['MCParticles.momentum.y'][mask][event_index]
        momentum_z = event_data['MCParticles.momentum.z'][mask][event_index]
        mass = event_data['MCParticles.mass'][mask][event_index]

        if self.config.ROTATE_DATA:
            momentum_x, momentum_z = self._rotateY(momentum_x, momentum_z, .025)
        momentum = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        theta = np.arccos(momentum_z/momentum)
        if self.config.USE_ETA_MIN or self.config.USE_ETA_MAX:
            # Doing this to avoid a warning from log(0).
            theta_np = ak.to_numpy(theta)
            tan_half = np.tan(theta_np / 2)
            eta = np.full_like(theta_np, np.inf, dtype=float)
            mask = tan_half > 0
            eta[mask] = -np.log(tan_half[mask])

        if self.config.THETA_UNITS == "mrad":
            theta = theta*1000
        overall_mask = np.ones_like(theta, dtype=bool)
        E_minus_pz_mask = np.ones_like(theta, dtype=bool)
        if self.config.USE_THETA_MAX:
            overall_mask = (overall_mask) & (theta < self.config.THETA_MAX)
            E_minus_pz_mask = (E_minus_pz_mask) & (theta < self.config.THETA_MAX)
        if self.config.USE_ETA_MIN:
            overall_mask = (overall_mask) & (eta > self.config.ETA_MIN)
            E_minus_pz_mask = (E_minus_pz_mask) & (eta > self.config.ETA_MIN)
        if self.config.USE_ETA_MAX:
            overall_mask = (overall_mask) & (eta < self.config.ETA_MAX)
            E_minus_pz_mask = (E_minus_pz_mask) & (eta < self.config.ETA_MAX)
        
        # If all particles removed, return an empty tuple
        if ~ak.any(overall_mask):
            return ()
        
        energy = np.sqrt(momentum**2 + mass**2)
        E_minus_pz = energy[E_minus_pz_mask] - momentum_z[E_minus_pz_mask]

        momentum_x = momentum_x[overall_mask]
        momentum_y = momentum_y[overall_mask]
        momentum_z = momentum_z[overall_mask]
        momentum = momentum[overall_mask]
        theta = theta[overall_mask]
        mass = mass[overall_mask]
        energy = energy[overall_mask]
        
        phi = np.arctan2(momentum_y, momentum_x)
        

        num_particles = len(momentum)
        regression_variables_to_values = {}
        if num_particles > 1:
            summed_momentum_x = ak.sum(momentum_x)
            summed_momentum_y = ak.sum(momentum_y)
            summed_momentum_z = ak.sum(momentum_z)

            total_momentum = np.sqrt(summed_momentum_x**2 + summed_momentum_y**2 + summed_momentum_z**2)
            log_momentum = np.log10(total_momentum)
            total_momentum_transverse = np.log10(np.sqrt(summed_momentum_x**2 + summed_momentum_y**2))
            total_E_minus_pz = np.log10(ak.sum(E_minus_pz))
            regression_variables_to_values["momentum"] = log_momentum
            regression_variables_to_values["transverse_momentum"] = total_momentum_transverse
            regression_variables_to_values["E_minus_pz"] = total_E_minus_pz
        elif num_particles == 1:
            momentum_transverse = np.sqrt(momentum_x**2 + momentum_y**2)
            momentum = momentum[0]
            log_momentum = np.log10(momentum)
            E_minus_pz = np.log10(E_minus_pz[0])
            momentum_transverse = np.log10(momentum_transverse[0])
            theta = theta[0]
            phi = phi[0]
            regression_variables_to_values["momentum"] = log_momentum
            regression_variables_to_values["transverse_momentum"] = momentum_transverse
            regression_variables_to_values["E_minus_pz"] = E_minus_pz
            regression_variables_to_values["theta"] = theta
            regression_variables_to_values["phi"] = phi
        output_tuple = ()
        for variable in self.config.REGRESSION_VARIABLES:
            variable_value = regression_variables_to_values[variable]
            normalized_value = (variable_value - self.means_dict[variable]) / self.stdvs_dict[variable]
            
            output_tuple += (normalized_value, )
        return output_tuple

    def _get_momentum(self, event_data, event_index, particle_name) -> np.ndarray:
        mask = self.mask_function(event_data, particle_name)

        momentum_x = event_data['MCParticles.momentum.x'][mask][event_index, 0]
        momentum_y = event_data['MCParticles.momentum.y'][mask][event_index, 0]
        momentum_z = event_data['MCParticles.momentum.z'][mask][event_index, 0]

        if self.config.ROTATE_DATA:
            momentum_x, momentum_z = self._rotateY(momentum_x, momentum_z, .025)

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]

        return momentum


    def _get_momentum_theta(self, event_data, event_index, particle_name) -> Tuple[any, any]:
        mask = self.mask_function(event_data, particle_name)

        momentum_x = event_data['MCParticles.momentum.x'][mask][event_index, 0]
        momentum_y = event_data['MCParticles.momentum.y'][mask][event_index, 0]
        momentum_z = event_data['MCParticles.momentum.z'][mask][event_index, 0]
        
        if self.config.ROTATE_DATA:
            momentum_x, momentum_z = self._rotateY(momentum_x, momentum_z, .025)

        momentum = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        theta = np.arccos(momentum_z/momentum)
        if self.config.THETA_UNITS == "mrad":
            theta = theta*1000

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]

        return momentum, theta


    def _get_momentum_theta_phi(self, event_data, event_index, particle_name) -> Tuple[any, any, any]:
        mask = self.mask_function(event_data, particle_name)
        momentum_x = event_data['MCParticles.momentum.x'][mask][event_index, 0]
        momentum_y = event_data['MCParticles.momentum.y'][mask][event_index, 0]
        momentum_z = event_data['MCParticles.momentum.z'][mask][event_index, 0]

        if self.config.ROTATE_DATA:
            momentum_x, momentum_z = self._rotateY(momentum_x, momentum_z, .025)

        momentum = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        theta = np.arccos(momentum_z/momentum)
        if self.config.THETA_UNITS == "mrad":
            theta = theta*1000
        phi = np.arctan2(momentum_y, momentum_x)

        momentum = np.log10(np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2))
        momentum = (momentum - self.means_dict["momentum"]) / self.stdvs_dict["momentum"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        phi = (phi -self.means_dict["phi"]) / self.stdvs_dict["phi"]

        return momentum, theta, phi


    def _get_particle_type(self, particle_name) -> int:
        if particle_name == self.config.PARTICLE0:
            return 0
        elif particle_name == self.config.PARTICLE1:
            return 1
        else:
            raise DataPreprocessorException(
                "Particle name doesn't match config particle names"
            )

    def _get_meta(self, event_data, event_index) -> list:
        """ 
        Reading meta data
        Returns senders, receivers, and edges    
        """ 
        meta_data = [] 
        meta_data.append(event_index)

        return meta_data
