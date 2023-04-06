import numpy as np
import os
import sys
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import argparse
import yaml

from generators import MPGraphDataGenerator
import block as models
sns.set_context('poster')

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

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
    already_preprocessed = data_config['already_preprocessed']
    calc_stats = data_config['calc_stats']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    save_dir = train_config['save_dir'] + '/Block_'+time.strftime("%Y%m%d_%H%M")+'_concat'+str(concat_input)
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    # print('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))

    root_files = np.sort(glob.glob(data_dir+'*root'))
    train_start = 0
    train_end = train_start + num_train_files
    val_end = train_end + num_val_files
    test_end = val_end + num_test_files

    # Comment the block below for real running
    train_end = train_start + 20
    val_end = train_end + 5
    test_end = val_end + 5

    root_train_files = root_files[train_start:train_end]
    root_val_files = root_files[train_end:val_end]
    root_test_files = root_files[val_end:test_end]

    print("\n\n Train Files = ",root_train_files)
    print("\n\n Val Files = ",root_val_files)
    print("\n\n Test Files = ",root_test_files)

    train_output_dir = None
    val_output_dir = None
            
    # Get Data
    if preprocess:
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/val/'

    data_gen_train = MPGraphDataGenerator(file_list=root_train_files,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_procs=num_procs,
                                          calc_stats=calc_stats,
                                          is_val=False,
                                          preprocess=preprocess,
                                          already_preprocessed=already_preprocessed,
                                          output_dir=train_output_dir)

    data_gen_val = MPGraphDataGenerator(file_list=root_val_files,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        calc_stats=calc_stats,
                                        is_val=True,
                                        preprocess=preprocess,
                                        already_preprocessed=already_preprocessed,
                                        output_dir=val_output_dir)

    data_gen_test = MPGraphDataGenerator(file_list=root_test_files,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        calc_stats=calc_stats,
                                        is_val=True, #decides to save mean and std
                                        preprocess=preprocess,
                                        already_preprocessed=already_preprocessed,
                                        output_dir=val_output_dir)

    # if preprocess and not already_preprocessed:
    #    exit()

    # Optimizer.
    #optimizer = snt.optimizers.Adam(learning_rate)
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
    if os.path.exists(last_ckpt_path+'.index'):
        checkpoint.read(last_ckpt_path)
    else:
        checkpoint.write(last_ckpt_path)

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

    samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
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

        return loss, predictions

    curr_loss = 1e5

    #Main Epoch Loop
    for e in range(epochs):

        print('\n\nStarting epoch: {}'.format(e))
        epoch_start = time.time()

        training_loss = []
        val_loss = []

        # Train
        print('Training...')
        i = 1
        start = time.time()
        for graph_data_tr, targets_tr in get_batch(data_gen_train.generator()):#train_iter):
            #if i==1:
            losses_tr = train_step(graph_data_tr, targets_tr)

            training_loss.append(losses_tr.numpy())

            if not (i)%50:
                end = time.time()
                print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                      format(i, training_loss[-1], np.mean(training_loss)), end='\t')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()
            
            i += 1 

        end = time.time()
        print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
              format(i, training_loss[-1], np.mean(training_loss)), end='\t')
        print('Took {:.3f} secs'.format(end-start))
    
        training_loss_epoch.append(training_loss)
        training_end = time.time()

        # validate
        print('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        all_etas = []
        start = time.time()
        for graph_data_val, targets_val in get_batch(data_gen_val.generator()):#val_iter):
            losses_val, output_vals = val_step(graph_data_val, targets_val)

            targets_val = targets_val.numpy()
            output_vals = output_vals.numpy().squeeze()

            val_loss.append(losses_val.numpy())
            all_targets.append(targets_val)
            all_outputs.append(output_vals)

            if not (i)%50:
                end = time.time()
                print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                      format(i, val_loss[-1], np.mean(val_loss)), end='\t')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()
            
            i += 1 

        end = time.time()
        print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
              format(i, val_loss[-1], np.mean(val_loss)), end='\t')
        print('Took {:.3f} secs'.format(end-start))

        epoch_end = time.time()

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        val_loss_epoch.append(val_loss)
    
        np.savez(save_dir+'/losses', training=training_loss_epoch, validation=val_loss_epoch)
        checkpoint.write(last_ckpt_path)

        val_mins = int((epoch_end - training_end)/60)
        val_secs = int((epoch_end - training_end)%60)
        training_mins = int((training_end - epoch_start)/60)
        training_secs = int((training_end - epoch_start)%60)
        print('\nEpoch {} ended\nTraining: {:2d}:{:02d}\nValidation: {:2d}:{:02d}'. \
              format(e, training_mins, training_secs, val_mins, val_secs))
    
        if np.mean(val_loss)<curr_loss:
            print('\nLoss decreased from {:.6f} to {:.6f}'.format(curr_loss, np.mean(val_loss)))
            print('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs)
            checkpoint.save(best_ckpt_prefix)
        else:
            print('\nLoss did not decrease from {:.6f}'.format(curr_loss))

        if not (e+1)%4:
            optimizer.learning_rate = optimizer.learning_rate/2
            if optimizer.learning_rate<1e-6:
                optimizer.learning_rate = 1e-6 
                print('\nLearning rate would fall below 1e-6, setting to: {:.5e}'.format(optimizer.learning_rate.value()))
            else:
                print('\nLearning rate decreased to: {:.5e}'.format(optimizer.learning_rate.value()))


#Inference over Test Dataset
print('\nTest Predictions...')
i = 1
test_loss = []
all_targets = []
all_outputs = []
all_etas = []
start = time.time()
for graph_data_test, targets_test in get_batch(data_gen_test.generator()):#test_iter):
    losses_test, output_tests = val_step(graph_data_test, targets_test) 
    # val_step above simply evaluates the model with the inputs as the first argument. Returns predictions.
    # This function is used for validation, but since THIS use block is outside of the training loop, 
    # the resulting loss and set of predictions are not used in the training at all.

    targets_test = targets_test.numpy()
    output_tests = output_tests.numpy().squeeze()

    test_loss.append(losses_test.numpy())
    all_targets.append(targets_test)
    all_outputs.append(output_tests)


    if not (i)%50: #FIXME: WTH is this loop for if not training?
        end = time.time()
        print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
              format(i, test_loss[-1], np.mean(test_loss)), end='\t')
        print('Took {:.3f} secs'.format(end-start))
        start = time.time()

    i += 1 

end = time.time()
print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
      format(i, test_loss[-1], np.mean(test_loss)), end='\t')
print('Took {:.3f} secs'.format(end-start))

epoch_end = time.time()

all_targets = np.concatenate(all_targets)
all_outputs = np.concatenate(all_outputs)

np.savez(save_dir+'/test_predictions', 
         targets=all_targets, 
         outputs=all_outputs)
np.savez(save_dir+'/test_loss', test=test_loss)


