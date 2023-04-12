import numpy as np
import os
import sys
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import argparse
import yaml

from generators import MPGraphDataGenerator
import block as models

test_loss = []
all_targets = []
all_outputs = []
all_etas = []
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='results/Block_20230407_1248_concatTrue')
# parser.add_argument('--config', default='configs/default.yaml')
args = parser.parse_args()

print("RESULTS DIR = ",args.results_dir)
print(f"CONFIG = {args.results_dir}/config.yaml")

config = yaml.safe_load(open(f'{args.results_dir}/config.yaml'))

save_dir = args.results_dir

data_config = config['data']
model_config = config['model']
train_config = config['training']

data_dir = data_config['data_dir']

num_train_files = data_config['num_train_files']
num_val_files = data_config['num_val_files']
num_test_files = data_config['num_test_files']
batch_size = data_config['batch_size']
shuffle = data_config['shuffle']
num_procs = data_config['num_procs']
preprocess = data_config['preprocess']
output_dir = data_config['output_dir']
# already_preprocessed = data_config['already_preprocessed']
already_preprocessed = True
calc_stats = data_config['calc_stats']
calc_stats = False

concat_input = model_config['concat_input']

epochs = train_config['epochs']
num_features=train_config['num_features']
learning_rate = train_config['learning_rate']
# save_dir = train_config['save_dir'] + '/Block_'+time.strftime("%Y%m%d_%H%M")+'_concat'+str(concat_input)
# os.makedirs(save_dir, exist_ok=True)
yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

# print('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))

root_files = np.sort(glob.glob(data_dir+'*root'))
train_start = 0
train_end = train_start + num_train_files
val_end = train_end + num_val_files
test_end = val_end + num_test_files

root_train_files = root_files[train_start:train_end]
root_val_files = root_files[train_end:val_end]
root_test_files = root_files[val_end:test_end]

print("\n\n Test Files = ",root_test_files)

# Get Data
if preprocess:
    train_output_dir = output_dir + '/train/'
    val_output_dir = output_dir + '/val/'
    test_output_dir = output_dir + '/test/'


data_gen_test = MPGraphDataGenerator(file_list=root_test_files,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_procs=num_procs,
                                     calc_stats=calc_stats,
                                     is_val=True,
                                     preprocess=preprocess,
                                     already_preprocessed=already_preprocessed,
                                     output_dir=test_output_dir,
                                     num_features=num_features)

optimizer = tf.keras.optimizers.Adam(learning_rate)

model = models.BlockModel(global_output_size=1, model_config=model_config)

training_loss_epoch = []
val_loss_epoch = []

## Checkpointing 
checkpoint = tf.train.Checkpoint(module=model)
best_ckpt_prefix = os.path.join(save_dir, 'best_model')
best_ckpt = tf.train.latest_checkpoint(save_dir)
last_ckpt_path = save_dir + '/last_saved_model'

if best_ckpt is not None:
    checkpoint.restore(best_ckpt)
# if os.path.exists(last_ckpt_path+'.index'):
#     checkpoint.read(last_ckpt_path)
# else:
#     checkpoint.write(last_ckpt_path)

#FIXME: move function definitions to something like `graph_functions.py`
#       then import them here, and in train_block.py
def convert_to_tuple(graphs):
    nodes = []
    edges = []
    globals = []
    senders = []
    receivers = []
    n_node = []
    n_edge = []
    offset = 0

    for graph in graphs:
        nodes.append(graph['nodes'])
        globals.append([graph['globals']])
        n_node.append(graph['nodes'].shape[:1])

        if graph['senders']:
            senders.append(graph['senders'] + offset)
        if graph['receivers']:
            receivers.append(graph['receivers'] + offset)
        if graph['edges']:
            edges.append(graph['edges'])
            n_edge.append(graph['edges'].shape[:1])
        else:
            n_edge.append([0])

        offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes))
        globals = tf.convert_to_tensor(np.concatenate(globals))
        n_node = tf.convert_to_tensor(np.concatenate(n_node))
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge))

        if senders:
            senders = tf.convert_to_tensor(np.concatenate(senders))
        else:
            senders = tf.convert_to_tensor(senders)
        if receivers:
            receivers = tf.convert_to_tensor(np.concatenate(receivers))
        else:
            receivers = tf.convert_to_tensor(receivers)
        if edges:
            edges = tf.convert_to_tensor(np.concatenate(edges))
        else:
            edges = tf.convert_to_tensor(edges)
            edges = tf.reshape(edges, (-1, 1))

        graph = GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globals,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge
        )

        return graph

def get_batch(data_iter):
    for graphs, targets, meta in data_iter:
        graphs = convert_to_tuple(graphs)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        yield graphs, targets

samp_graph, samp_target = next(get_batch(data_gen_test.generator()))
data_gen_test.kill_procs()
graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

mae_loss = tf.keras.losses.MeanAbsoluteError()

def loss_fn(targets, predictions):
    return mae_loss(targets, predictions) 

@tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
def train_step(graphs, targets):
    with tf.GradientTape() as tape:
        predictions = model(graphs).globals
        loss = loss_fn(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

@tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
def val_step(graphs, targets):
    predictions = model(graphs).globals
    loss = loss_fn(targets, predictions)
    print("IN val_step: ",predictions)

    return loss, predictions

    curr_loss = 1e5

#Inference over Test Dataset
print('\nTest Predictions...')
i = 1

for graph_data_test, targets_test in get_batch(data_gen_test.generator()): #test_iter
    losses_test, output_tests = val_step(graph_data_test, targets_test) 

    targets_test = targets_test.numpy()
    output_tests = output_tests.numpy().squeeze()

    test_loss.append(losses_test.numpy())
    all_targets.append(targets_test)
    all_outputs.append(output_tests)

    print("TARGET = ",targets_test)
    print("OUTPUT = ",output_tests)

    if not (i)%50:
        end = time.time()
        print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
              format(i, test_loss[-1], np.mean(test_loss)), end='\t')
        print('Took {:.3f} secs'.format(end-start))
        start = time.time()

        i += 1 

    end = time.time()
    print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
          format(i, test_loss[-1], np.mean(test_loss)), end='\t')
    print('Took {:.3f} secs'.format(end-start))

    epoch_end = time.time()

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    np.savez(save_dir+'/test_predictions', 
             targets=all_targets, 
             outputs=all_outputs)
    np.savez(save_dir+'/test_loss', test=test_loss)


