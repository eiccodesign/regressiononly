from pathlib      import Path
import yaml
import numpy as np
from exceptions   import ConfigLoaderException


class ConfigLoader:

    """
    This class is responsible for loading configuration settings from YAML 
    files located in the 'configs' directory. Upon initialization, it reads 
    the specified files to set up various configurations such as data, model,
    and training settings, making these settings available as attributes of 
    the class instance. This allows for easy access and management of 
    configuration parameters throughout the application.
    """

    def __init__(self, configs_folder_path: str = 'configs'):
        current_file = Path(__file__).parent.resolve()
        self.CONFIGS_FOLDER_PATH =  current_file / configs_folder_path 
        print(self.CONFIGS_FOLDER_PATH)

        if (not self.CONFIGS_FOLDER_PATH.exists() or 
            not self.CONFIGS_FOLDER_PATH.is_dir()):
            raise ConfigLoaderException(
                "The specified configuration directory does not exist or is " 
                "not a directory."
            )
      
        self._load_data_configs()
        self._load_model_configs()
        self._load_training_configs()    


    """
    The below public functions return the entire yaml file as a python 
    dictionary. Allowing for more flexibility in how the configs are
    accessed. 
    """

    def get_data_configs(self) -> dict:
        file_path = self.CONFIGS_FOLDER_PATH / 'data.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("data.yaml file does not exist")
        
        with file_path.open('r') as file:
            data_configs = yaml.safe_load(file)
            return data_configs


    def get_model_configs(self) -> dict:
        file_path = self.CONFIGS_FOLDER_PATH / 'model.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("model.yaml file does not exist")
        
        with file_path.open('r') as file:
            model_configs = yaml.safe_load(file)
            return model_configs
        

    def get_training_configs(self) -> dict:
        file_path = self.CONFIGS_FOLDER_PATH / 'training.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("training.yaml file does not exist")
        
        with file_path.open('r') as file:
            training_configs = yaml.safe_load(file)
            return training_configs
    

    """
    Prints out all currently used configs to console when called
    """

    def print_configs(self):
        print("Configuration Settings:")
        for attr, value in self.__dict__.items():
            print(f"    {attr}: {value}")


    """
    Private class methods called when the class is instantiated and loads
    all config settings as class variables which can be access with
    config."variable".
    """

    def _load_data_configs(self):
        file_path = self.CONFIGS_FOLDER_PATH / 'data.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("data.yaml does not exist")
        
        with file_path.open('r') as file:
            data_configs = yaml.safe_load(file)
            for key, value in data_configs.items():
              setattr(self, key, value)
        
        self.detector_dictionary = {
            "insert" : {
                "BRANCH_NAME" : "HcalEndcapPInsertRecHits",
                "SAMPLING_FRACTION" : .02,
                "ENERGY_TH" : 0.5 * 0.0006,
                "TIME_TH" : 150,
                "THETA_MAX" : 10000000, # in radians
                "DETECTOR_TYPE" : "HCAL"
            },
            "LFHCAL" : {
                "BRANCH_NAME" : "LFHCALRecHits",
                "SAMPLING_FRACTION" : 1,
                "ENERGY_TH" : None,
                "TIME_TH" : 150,
                "THETA_MAX" : 10000000, # in radians
                "DETECTOR_TYPE" : "HCAL"
            },
            "hcal" : {
                "BRANCH_NAME" : "HcalEndcapPHitsReco",
                "SAMPLING_FRACTION" : .0224,
                "ENERGY_TH" : 0.5 * 0.0006,
                "TIME_TH" : 150,
                "THETA_MAX" : 1000, # in radians
                "DETECTOR_TYPE" : "HCAL"
            },
            "zdc_Fe" : {
                "BRANCH_NAME" : "ZDCHcalHitsReco",
                "SAMPLING_FRACTION" : .0203,
                "ENERGY_TH" : 0.5 * 0.000472,
                "TIME_TH" : 275,
                "THETA_MAX" : 10, # in radians
                "DETECTOR_TYPE" : "HCAL"
            },
            "zdc_Pb" : {
                "BRANCH_NAME" : "ZDCHcalHitsReco",
                "SAMPLING_FRACTION" : .0216,
                "ENERGY_TH" : 0.5 * 0.000393,
                "TIME_TH" : 275,
                "THETA_MAX" : 4, # in radians
                "DETECTOR_TYPE" : "HCAL"
            },
            "zdc_ecal" : {
                "BRANCH_NAME" : 'ZDCEcalHitsReco',
                "SAMPLING_FRACTION" : 1,
                "ENERGY_TH" : 0.5 * .088,
                "TIME_TH" : None,
                "THETA_MAX" : 4, # in radians
                "DETECTOR_TYPE" : "ECAL"
            },
            "ecal_insert" : {
                "BRANCH_NAME" : 'EcalEndcapPInsertRecHits',
                "SAMPLING_FRACTION" : 1,
                "ENERGY_TH" : .05 * 0.13,
                "TIME_TH" : None,
                "THETA_MAX" : 100000000, # in radians
                "DETECTOR_TYPE" : "ECAL"
            },
            "ecal" : {
                "BRANCH_NAME" : 'EcalEndcapPRecHits',
                "SAMPLING_FRACTION" : 1,
                "ENERGY_TH" : .05 * 0.13,
                "TIME_TH" : None,
                "THETA_MAX" : 100000000, # in radians
                "DETECTOR_TYPE" : "ECAL"
            },
            "muon_detector" : {
                "BRANCH_NAME" : 'MuographyHits',
                "SAMPLING_FRACTION" : 1,
                "ENERGY_TH" : None,
                "TIME_TH" : None,
                "THETA_MAX" : None,
                "DETECTOR_TYPE" : "HCAL"
            }
        }

        hcal_names = self.HCAL_NAMES
        ecal_names = self.ECAL_NAMES

        for ecal_name in ecal_names:
            if ecal_name not in self.detector_dictionary:
                raise ConfigLoaderException(
                    f"Invalid name in ECAL_NAMES in data.yaml: {ecal_name}"
                )
        for hcal_name in hcal_names:
            if hcal_name not in self.detector_dictionary:
                raise ConfigLoaderException(
                    f"Invalid name in HCAL_NAMES in data.yaml: {hcal_name}"
                )
        
        self.detector_names = hcal_names + ecal_names
        if self.THETA_UNITS == "mrad":
            for detector in self.detector_names:
                if self.detector_dictionary[detector]["THETA_MAX"] is not None:
                    self.detector_dictionary[detector]["THETA_MAX"] *= 1000
        elif self.THETA_UNITS == "deg":
            for detector in self.detector_names:
                if self.detector_dictionary[detector]["THETA_MAX"] is not None:
                    self.detector_dictionary[detector]["THETA_MAX"] *= (180/np.pi)

        self.NODE_FEATURE_NAMES = [
            ".energy", 
            ".position.z", 
            ".position.x",
            ".position.y", 
        ]

        self.EDGE_FEATURE_NAMES = [
            ".position.z", 
            ".position.x",
            ".position.y", 
        ]

        valid_regression_variables = [
            "momentum",
            "theta",
            "phi",
            "transverse_momentum",
            "E_minus_pz"
        ]
        bad_regression_variables = [variable for variable in self.REGRESSION_VARIABLES if variable not in valid_regression_variables]
        if len(bad_regression_variables) > 0:
            raise ValueError(
                f"Unsupported variables in REGRESSION_VARIABLES: {bad_regression_variables}. "
                f"The supported regression variables are {valid_regression_variables}. "
                "Please only use the supported regression variables in configs/data.yaml"
            )

        self.regression_variable_to_output_index = {}
        for i, variable in enumerate(self.REGRESSION_VARIABLES):
            self.regression_variable_to_output_index[variable] = i
        self.NUM_NODE_FEATURES = len(self.NODE_FEATURE_NAMES)
        
        self.SCALAR_KEYS = []
        for detector_name in self.detector_names:
            self.SCALAR_KEYS += [self.detector_dictionary[detector_name]["BRANCH_NAME"] + ".energy"]
        self.SCALAR_KEYS += self.NODE_FEATURE_NAMES[1:]
        self.SCALAR_KEYS += ["cluster_energy"]
        self.SCALAR_KEYS += self.REGRESSION_VARIABLES
        self.REGRESSION_OUTPUT_DIMENSIONS = len(self.REGRESSION_VARIABLES)


        if self.USE_CLASSIFICATION is True:
            self.REGRESSION_WEIGHT = self.CLASSIFICATION_SETTINGS["REGRESSION_WEIGHT"]
            self.CLASSIFICATION_WEIGHT = self.CLASSIFICATION_SETTINGS["CLASSIFICATION_WEIGHT"]
            self.PARTICLE0 = self.CLASSIFICATION_SETTINGS["PARTICLE0"]
            self.PARTICLE1 = self.CLASSIFICATION_SETTINGS["PARTICLE1"]
            self.PARTICLE0_TRAINING_DATA_PATH = Path(self.CLASSIFICATION_SETTINGS["PARTICLE0_TRAINING_DATA_PATH"])
            self.PARTICLE0_TEST_DATA_PATH = Path(self.CLASSIFICATION_SETTINGS["PARTICLE0_TEST_DATA_PATH"])
            self.PARTICLE1_TRAINING_DATA_PATH = Path(self.CLASSIFICATION_SETTINGS["PARTICLE1_TRAINING_DATA_PATH"])
            self.PARTICLE1_TEST_DATA_PATH = Path(self.CLASSIFICATION_SETTINGS["PARTICLE1_TEST_DATA_PATH"])
        else:
            self.TRAINING_DATA_PATH = Path(self.NO_CLASSIFICATION_SETTINGS["TRAINING_DATA_PATH"])
            self.TEST_DATA_PATH = Path(self.NO_CLASSIFICATION_SETTINGS["TEST_DATA_PATH"])
            self.PARTICLE = self.NO_CLASSIFICATION_SETTINGS["PARTICLE"]
        self.OUTPUT_DIR_PATH = Path(self.OUTPUT_DIR_PATH)    

        if self.PREPROCESS_DATA is True:
            self.TRAIN_OUTPUT_DIR_PATH =  self.OUTPUT_DIR_PATH / 'train/'
            self.VAL_OUTPUT_DIR_PATH =  self.OUTPUT_DIR_PATH / 'val/'

    
    def _load_model_configs(self):
        file_path = self.CONFIGS_FOLDER_PATH / 'model.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("model.yaml file does not exist")
        
        with file_path.open('r') as file:
            model_configs = yaml.safe_load(file)
            
            for key, value in model_configs.items():
                setattr(self, key, value)

        
    def _load_training_configs(self):
        file_path = self.CONFIGS_FOLDER_PATH / 'training.yaml'

        if not file_path.exists():
            raise ConfigLoaderException("training.yaml file does not exist")
        
        with file_path.open('r') as file:
            training_configs = yaml.safe_load(file)
            for key, value in training_configs.items():
              setattr(self, key, value)