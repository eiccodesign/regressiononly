from glob                       import glob
from sklearn.model_selection    import train_test_split

import numpy                    as np
import tensorflow               as tf
import block                    as external_models
import logging
import os

from config_loader              import ConfigLoader
from data_preprocessor          import DataPreprocessor
from data_generator             import DataGenerator
from data_normalizer            import DataNormalizer
from model                      import Model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all logs except for fatal errors
logging.getLogger('tensorflow').setLevel(logging.FATAL)


config = ConfigLoader('configs')

root_files = glob(str(config.TEST_DATA_PATH / '*.root'))
root_files = np.sort(root_files)

def mask_function(event_data):
    return event_data["MCParticles.generatorStatus"] == 1

normalizer = DataNormalizer(config, root_files, "val")
test_data = DataPreprocessor(config, root_files, "test", mask_function)


graph_net_model = external_models.BlockModel(
    global_output_size=config.OUTPUT_DIMENSIONS, 
    model_config=config.get_model_configs()
)


model = Model(
    config_loader=config, 
    model=graph_net_model,
)



checkpoint = tf.train.Checkpoint(module=graph_net_model)
best_ckpt_prefix = os.path.join(config.RESULT_DIR_PATH, '/best_model')
best_ckpt = tf.train.latest_checkpoint(config.RESULT_DIR_PATH)
last_ckpt_path = config.RESULT_DIR_PATH + '/last_saved_model'


means_dict, stdvs_dict = normalizer.get_normalizer_dicts()
test_data = DataGenerator(config, root_files, "test")

all_targets_scaled, all_outputs_scaled, all_targets, all_outputs, all_meta = model.get_pred_3D(test_data, means_dict, stdvs_dict)

all_targets = np.concatenate(all_targets)
all_outputs = np.concatenate(all_outputs)
all_meta = np.concatenate(all_meta)

print(f"\n Done. Completed {np.shape(all_targets)}\n")
np.savez(config.RESULT_DIR_PATH+'/predictions_appended_test.npz',
            targets=all_targets, targets_scaled=all_targets_scaled,
            outputs=all_outputs, outputs_scaled=all_outputs_scaled,
            meta=all_meta)