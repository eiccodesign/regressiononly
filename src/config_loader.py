from pathlib      import Path
import yaml

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

        if self.HADRONIC_DETECTOR == 'hcal':
            self.DETECTOR_NAME     = "HcalEndcapPHitsReco"
            self.SAMPLING_FRACTION = 0.0224
            self.ENERGY_TH         = 0.5 * 0.0006
            self.TIME_TH           = 150
            self.THETA_MAX         = 1000.0
        elif self.HADRONIC_DETECTOR == 'insert':
            self.DETECTOR_NAME     = "HcalEndcapPInsertHitsReco"
            self.SAMPLING_FRACTION = 0.0089
            self.ENERGY_TH         = 0.5 * 0.0006
            self.TIME_TH           = 150
            self.THETA_MAX         = 76.0
        elif self.HADRONIC_DETECTOR == 'zdc_Fe':
            self.DETECTOR_NAME     = "ZDCHcalHitsReco"
            self.SAMPLING_FRACTION = 0.0203
            self.ENERGY_TH         = 0.5 * 0.000472
            self.TIME_TH           = 275
            self.THETA_MAX         = 10.0 
        elif self.HADRONIC_DETECTOR == 'zdc_Pb':
            self.DETECTOR_NAME     = "ZDCHcalHitsReco"
            self.SAMPLING_FRACTION = 0.0216
            self.ENERGY_TH         = 0.5 * 0.000393
            self.TIME_TH           = 275
            self.THETA_MAX         = 4.0
        else:
            raise ConfigLoaderException(
                "Invalid hadronic_detector argument in data.yaml"
            )
        
        self.DETECTOR_ECAL = 'EcalEndcapPHitsReco'
        self.MIP_ECAL = 0.13
        self.ENERGY_TH_ECAL = 0.5 * self.MIP_ECAL

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

        self.NUM_NODE_FEATURES = len(self.NODE_FEATURE_NAMES)

        if self.OUTPUT_DIMENSIONS == 1: # Energy regression only
            self.SCALAR_KEYS = [
                self.DETECTOR_NAME + ".energy",
                *self.NODE_FEATURE_NAMES[1:],
                "cluster_energy",
                "momentum",
            ]
        elif self.OUTPUT_DIMENSIONS == 2:  # Energy + theta regression
            self.SCALAR_KEYS = [
                self.DETECTOR_NAME + ".energy",
                *self.NODE_FEATURE_NAMES[1:],
                "cluster_energy",
                "momentum",
                "theta",
            ]
        elif self.OUTPUT_DIMENSIONS == 3: # Energy + theta + phi regression
            self.SCALAR_KEYS = [
                self.DETECTOR_NAME + ".energy",
                *self.NODE_FEATURE_NAMES[1:],
                "cluster_energy",
                "momentum",
                "theta",
                "phi"
            ]
        else:
            raise ConfigLoaderException(
                "Invalid output_dimension argument in data.yaml"
            )

        if self.INCLUDE_ECAL is True:
            self.SCALAR_KEYS += [self.DETECTOR_ECAL + self.NODE_FEATURE_NAMES]  

        self.TRAINING_DATA_PATH = Path(self.TRAINING_DATA_PATH)
        self.NUM_TRAINING_FILES = sum(
            1 for file in self.TRAINING_DATA_PATH.iterdir() if file.is_file()
        )

        self.TEST_DATA_PATH = Path(self.TEST_DATA_PATH)
        self.NUM_TEST_FILES = sum(
            1 for file in self.TEST_DATA_PATH.iterdir() if file.is_file()
        )
        
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