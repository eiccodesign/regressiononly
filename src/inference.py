from glob                       import glob
from sklearn.model_selection    import train_test_split

import numpy                    as np
import tensorflow               as tf
import block                    as external_models
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from config_loader              import ConfigLoader
from data_preprocessor          import DataPreprocessor
from data_generator             import DataGenerator
from data_normalizer            import DataNormalizer
from model                      import Model

import tf2onnx
import onnx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all logs except for fatal errors
logging.getLogger('tensorflow').setLevel(logging.FATAL)


config = ConfigLoader('configs')

def mask_function(event_data, particle_name):
    if particle_name == "lambda":
        return (event_data["MCParticles.generatorStatus"] == 2) & (event_data["MCParticles.PDG"]==3122)
    elif particle_name == "sigma":
        return (event_data["MCParticles.generatorStatus"] == 2) & (event_data["MCParticles.PDG"]==3212)
    else:
        return event_data["MCParticles.generatorStatus"] == 1

if config.USE_CLASSIFICATION:
    particle0_root_files = glob(str(config.PARTICLE0_TEST_DATA_PATH / '*.root'))
    particle0_root_files = np.sort(particle0_root_files)
    particle0_labels = np.full(len(particle0_root_files), config.PARTICLE0)
    particle0_root_files = list(zip(particle0_root_files, particle0_labels))
    particle1_root_files = glob(str(config.PARTICLE1_TEST_DATA_PATH / '*.root'))
    particle1_root_files = np.sort(particle1_root_files)
    particle1_labels = np.full(len(particle1_root_files), config.PARTICLE1)
    particle1_root_files = list(zip(particle1_root_files, particle1_labels))
    root_files = np.concatenate((particle0_root_files, particle1_root_files))
else:
    root_files = glob(str(config.TEST_DATA_PATH / '*.root'))
    particle_labels = np.full(len(root_files), config.PARTICLE)
    root_files = list(zip(root_files, particle_labels))

root_files = np.sort(root_files)
config.CALC_NORMALIZER_STATS = False
normalizer = DataNormalizer(config, root_files, "val", mask_function)
test_data = DataPreprocessor(config, root_files, "test", mask_function)

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



checkpoint = tf.train.Checkpoint(module=graph_net_model)
best_ckpt_prefix = os.path.join(config.RESULT_DIR_PATH, '/best_model')
best_ckpt = tf.train.latest_checkpoint(config.RESULT_DIR_PATH)
last_ckpt_path = config.RESULT_DIR_PATH + '/last_saved_model'
if os.path.exists(best_ckpt+'.index'):
    print(f'Restoring {best_ckpt}')
    checkpoint.restore(best_ckpt)
else:
    print("\nCould not load best checkpoint. EXITING\n")
    exit()


means_dict, stdvs_dict = normalizer.get_normalizer_dicts()
test_data = DataGenerator(config, root_files, "test")
all_targets_scaled_dict, all_outputs_scaled_dict, all_targets_dict, all_outputs_dict, all_meta = model.get_predictions(test_data, means_dict, stdvs_dict)

print(f"\n Done.")
np.savez(config.RESULT_DIR_PATH+'/predictions_appended_test.npz',
            targets=all_targets_dict, targets_scaled=all_targets_scaled_dict,
            outputs=all_outputs_dict, outputs_scaled=all_outputs_scaled_dict,
            meta=all_meta)
        
input_signature = model._get_input_signature(test_data)
@tf.function(input_signature=[input_signature[0]])
def get_model(x):
    return graph_net_model(x)
# tf2onnx
model_proto, _ = tf2onnx.convert.from_function(
    get_model,
    input_signature=[input_signature[0]], opset=None, custom_ops=None,
    custom_op_handlers=None, custom_rewriter=None,
    inputs_as_nchw=None, extra_opset=None, shape_override=None,
    target=None, large_model=False, output_path=config.RESULT_DIR_PATH+"/gnn.onnx")