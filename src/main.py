from glob                       import glob
from sklearn.model_selection    import train_test_split

import numpy                    as np
import tensorflow               as tf
import block                    as external_models
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from config_loader              import ConfigLoader
from data_preprocessor          import DataPreprocessor
from data_generator             import DataGenerator
from data_normalizer            import DataNormalizer
from model                      import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all logs except for fatal errors
logging.getLogger('tensorflow').setLevel(logging.FATAL)

config = ConfigLoader('configs')

def mask_function(event_data, particle_name):
    if particle_name == "lambda":
        return (event_data["MCParticles.generatorStatus"] == 2) & (event_data["MCParticles.PDG"]==3122)
    elif particle_name == "sigma":
        return (event_data["MCParticles.generatorStatus"] == 2) & (event_data["MCParticles.PDG"]==3212)
    elif particle_name == "rho":
        return (event_data["MCParticles.generatorStatus"] == 2) & (event_data["MCParticles.PDG"]==113)
    else:
        return event_data["MCParticles.generatorStatus"] == 1

if config.USE_CLASSIFICATION:
    particle0_root_files = glob(str(config.PARTICLE0_TRAINING_DATA_PATH / '*.root'))
    particle0_root_files = np.sort(particle0_root_files)
    particle0_labels = np.full(len(particle0_root_files), config.PARTICLE0)
    particle0_root_files = list(zip(particle0_root_files, particle0_labels))
    num_particle0_root_files = len(particle0_root_files)
    particle1_root_files = glob(str(config.PARTICLE1_TRAINING_DATA_PATH / '*.root'))
    particle1_root_files = np.sort(particle1_root_files)
    particle1_labels = np.full(len(particle1_root_files), config.PARTICLE1)
    particle1_root_files = list(zip(particle1_root_files, particle1_labels))
    num_particle1_root_files = len(particle1_root_files)
    
    # Making sure that there are the same number of files for particle 1 and particle 2
    if num_particle0_root_files > num_particle1_root_files:
        particle0_root_files = particle0_root_files[:num_particle1_root_files]
    elif num_particle1_root_files > num_particle0_root_files:
        particle1_root_files = particle1_root_files[:num_particle0_root_files]
    if len(particle0_root_files)==0:
        print("Particle 0 has no data files!")
    if len(particle1_root_files)==0:
        print("Particle 1 has no data files!")

    particle0_train_files, particle0_val_files = train_test_split(
                                                                  particle0_root_files,
                                                                  train_size=config.TRAINING_FRACTION, 
                                                                  test_size=config.VALIDATION_FRACTION, 
                                                                  shuffle=config.SHUFFLE_FILES
                                                                 )
    particle1_train_files, particle1_val_files = train_test_split(
                                                                  particle1_root_files,
                                                                  train_size=config.TRAINING_FRACTION, 
                                                                  test_size=config.VALIDATION_FRACTION, 
                                                                  shuffle=config.SHUFFLE_FILES
                                                                 )
    train_files = np.concatenate((particle0_train_files, particle1_train_files))
    np.random.shuffle(train_files)
    val_files = np.concatenate((particle0_val_files, particle1_val_files))
    np.random.shuffle(val_files)

else:
    root_files = glob(str(config.TRAINING_DATA_PATH / '*.root'))
    root_files = np.sort(root_files)
    particle_labels = np.full(len(root_files), config.PARTICLE)
    root_files = list(zip(root_files, particle_labels))
    train_files, val_files = train_test_split(
        root_files,
        train_size=config.TRAINING_FRACTION, 
        test_size=config.VALIDATION_FRACTION, 
        shuffle=config.SHUFFLE_FILES
    )


normalizer = DataNormalizer(config, val_files, "val", mask_function)
val_data = DataPreprocessor(
    config, 
    val_files, 
    "val",
    mask_function
)
train_data = DataPreprocessor(
    config, 
    train_files, 
    "train",
    mask_function
)
if config.USE_CLASSIFICATION:
    model_output_size = config.REGRESSION_OUTPUT_DIMENSIONS + 1
else:
    model_output_size = config.REGRESSION_OUTPUT_DIMENSIONS
graph_net_model = external_models.BlockModel(
    global_output_size=model_output_size,
    model_config=config.get_model_configs()
)

model = Model(
    config_loader=config, 
    model=graph_net_model,
)


model.train_model(
    val_data=DataGenerator(config, val_files, "val"),
    train_data=DataGenerator(config, train_files, "train")
)



