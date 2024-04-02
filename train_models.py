import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from generator_common import MPGraphDataGenerator
import block as models
sns.set_context('poster')

# include HCAL and ECAL as in training_block.py

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_ucr.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']
    
    data_dir = data_config['data_dir']
    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    already_preprocessed = data_config['already_preprocessed']
    calc_stats = data_config['calc_stats']
    num_features = data_config['num_features']
    k = data_config['k']
    output_dim = data_config['output_dim']
    model_output_size = output_dim
    if output_dim == 2:
        energy_weight = data_config['energy_weight']
        theta_weight = data_config['theta_weight']
    hadronic_detector = data_config['hadronic_detector']
    include_ecal = data_config['include_ecal']
    use_classification = data_config['use_classification']
    if use_classification:
        regression_weight = data_config['regression_weight']
        classification_weight = data_config['classification_weight']
        model_output_size += 1
        num_pi0_val_files = data_config['num_pi0_val_files']
        num_photon_val_files = data_config['num_photon_val_files']
        num_pi0_train_files = data_config['num_pi0_train_files']
        num_photon_train_files = data_config['num_photon_train_files']
    
    #num_z_layers=data_config['num_z_layers']
    block_type = model_config['block_type']
    print('From the train block model ',num_features, 'output _dim ',  output_dim)
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']

    save_dir = train_config['save_dir'] + '/ECCE_'+time.strftime("%Y%m%d-%H%M_")+block_type+f'_{num_features:d}D'
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    root_files = glob.glob(data_dir+'*root')

    def list_filter(string_list, substring_list):
        return [str for str in string_list if any(sub in str for sub in substring_list)]
    if use_classification:
        pi0_filter = ['pi0']
        photon_filter = ['gamma']
        pi0_val_files = list_filter(root_files, pi0_filter)[:num_pi0_val_files]
        photon_val_files = list_filter(root_files, photon_filter)[:num_photon_val_files]
        root_val_files = pi0_val_files + photon_val_files
        np.random.shuffle(root_val_files)
        root_files = [i for i in root_files if i not in root_val_files]
        np.random.shuffle(root_files)
        train_start = 0
        train_end = train_start + num_pi0_train_files + num_photon_train_files
        root_train_files = root_files[train_start:train_end]

    else:
        root_files = np.sort(root_files)
        train_start = 0
        train_end = train_start + num_train_files
        val_end = train_end + num_val_files
        root_train_files = root_files[train_start:train_end]
        root_val_files = root_files[train_end:val_end]

    train_output_dir = None
    val_output_dir = None
    
    # Get Data
    if preprocess:
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/val/'

    #Generators
    data_gen_train = MPGraphDataGenerator(file_list=root_train_files,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_procs=num_procs,
                                          calc_stats=calc_stats, #calc_stats
                                          is_val=False,
                                          preprocess=preprocess,
                                          already_preprocessed=already_preprocessed,
                                          output_dir=train_output_dir,
                                          hadronic_detector=hadronic_detector,
                                          include_ecal=include_ecal,
                                          num_features=num_features,
                                          output_dim=output_dim,
                                          k=k,
                                          classification=use_classification)
                                          
        
    data_gen_val = MPGraphDataGenerator(file_list=root_val_files,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        calc_stats=False,
                                        is_val=True,
                                        preprocess=preprocess,
                                        already_preprocessed=already_preprocessed,
                                        output_dir=val_output_dir,
                                        hadronic_detector=hadronic_detector,
                                        include_ecal=include_ecal,
                                        num_features=num_features,
                                        output_dim=output_dim,
                                        k=k,
                                        classification=use_classification)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = models.BlockModel(global_output_size=model_output_size, model_config=model_config)

    training_loss_epoch = []
    val_loss_epoch = []

    ## Checkpointing 
    checkpoint = tf.train.Checkpoint(module=model)
    best_ckpt_prefix = os.path.join(save_dir, 'best_model') # prefix.

    best_ckpt = tf.train.latest_checkpoint(save_dir)
    last_ckpt_path = save_dir + '/last_saved_model'
    
    # if best_ckpt is not None:
    #     print(f'Restoring {best_ckpt}')
    #     checkpoint.restore(best_ckpt)
    if os.path.exists(last_ckpt_path+'.index'):
        print(f'Restoring {last_ckpt_path}')
        checkpoint.read(last_ckpt_path)
    else:
        print(f'Writing {last_ckpt_path}')
        checkpoint.write(last_ckpt_path)

    def convert_to_tuple(graphs):

        nodes = []
        edges = []
        globals = [] # may collide.

        senders = []
        receivers = []

        n_node = []
        n_edge = []

        offset = 0

        for graph in graphs:
            nodes.append(graph['nodes'])
            globals.append([graph['globals']])
            n_node.append(graph['nodes'].shape[:1])

            if graph['senders'] is not None:
                senders.append(graph['senders'] + offset)
            if graph['receivers'] is not None:
                receivers.append(graph['receivers'] + offset)
            if graph['edges'] is not None:
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
        print("In get_batch")
        for graphs, targets, meta in data_iter:
            # data_iter is a triple of lists
            # list of graphs, list of targets, list of meta data with
            # Each entry of the lists has info for one event in the batch
            graphs = convert_to_tuple(graphs)
            # targets structure: 
            # For 1D: Just a list [ genP0, genP1, genP2, ...]
            # For 2D: list of tuples [ (genP0, gentheta0), (genP1 ,Gentheta1),...]
            # Convert targets to tf.tensor
            # 1D shape (len(targets), ), i.e. [ genP0, genP1, genP2, ...]
            # 2D shape (len(targets), 2), i.e. [ [genP0, gentheta0], [genP1, gentheta1], ...]
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            yield graphs, targets, meta
    
    samp_graph, samp_target, samp_meta = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    
    mae_loss = tf.keras.losses.MeanAbsoluteError() # Check 
    classification_loss =  tf.keras.losses.BinaryCrossentropy()

    def mae_loss_fn(targets, predictions):
        if output_dim == 2:
            # Convert targets & predictions to tf.tensor of shape (len(targets), 2, 1)
            # i.e. Targets: [ [ [genP0], [gentheta0] ], [ [genP1], [gentheta1] ], [ [genP2], [gentheta2] ], ...]
            # Predictions: [ [ [Epred0], [thetapred0] ], [ [Epred1], [thetapred1] ], [ [Epred2], [thetapred2] ], ...]
            # This shape is needed to give weights to energy and theta. Gives prefactors to energy and theta.
            # e.g. Loss contribution for event 0:  ( energy_weight * |Epred0 - genP0|  + theta_weight * |thetapred0 - gentheta0 | ) / 2
            targets_reshaped = tf.reshape(targets, [len(targets), 2, 1])
            predictions_reshaped = tf.reshape(predictions, [len(predictions), 2, 1])
            return mae_loss(targets_reshaped, predictions_reshaped, sample_weight=[[energy_weight, theta_weight]]) # First number is energy weight, second number is theta weight
        elif output_dim == 1:
            return mae_loss(targets, predictions)
    def classification_loss_fn(targets, predictions):
        return classification_loss(targets, predictions)
    if output_dim==2:
        provided_shape=[None,None]
    elif output_dim==1:
        provided_shape=[None,]

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=provided_shape, dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            predictions = model(graphs).globals
            # predictions structure:
            # For 1D: predictions (type = tf.Tensor) structure: [ Epred0, Epred1, Epred2, ... ]
            # For 2D: predictions (type = tf.Tensor) structure: [ [Epred0, thetapred0], [Epred1, thetapred1], [Epred2, thetapred2], ...]
            # For 2D + classification: predictions (type = tf.Tensor) structure: [ [Epred0, thetapred0, ptype0], [Epred1, thetapred1, ptype1], [Epred2, thetapred2, ptype2], ...]
            if use_classification:
                # Classification info is stored at the back of the list
                predictions_regression, predictions_classification = predictions[:, :-1], predictions[:, -1:]
                targets_regression, targets_classification = targets[:, :-1], targets[:, -1:]
                loss = regression_weight*mae_loss_fn(targets_regression, predictions_regression) + classification_weight*classification_loss_fn(targets_classification, tf.math.sigmoid(predictions_classification))
            else:
                loss = mae_loss_fn(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=provided_shape, dtype=tf.float32)])
    def val_step(graphs, targets):
        predictions = model(graphs).globals
        if use_classification:
            # Classification info is stored at the back of the list
            predictions_regression, predictions_classification = predictions[:, :-1], predictions[:, -1:]
            targets_regression, targets_classification = targets[:, :-1], targets[:, -1:]
            loss = regression_weight*mae_loss_fn(targets_regression, predictions_regression) + classification_weight*classification_loss_fn(targets_classification, tf.math.sigmoid(predictions_classification))
        else:
            loss = mae_loss_fn(targets, predictions)

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
        i = 0 # 1 change
        start = time.time()
        for graph_data_tr, targets_tr, meta_tr in get_batch(data_gen_train.generator()):#train_iter):
            #if i==1:
            losses_tr = train_step(graph_data_tr, targets_tr)
            training_loss.append(losses_tr.numpy())

            if not (i)%100:
                end = time.time()
                print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                      format(i, training_loss[-1], np.mean(training_loss)), end='  ')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()

            i += 1 

        end = time.time()
        print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
              format(i, training_loss[-1], np.mean(training_loss)), end='  ')
        print('Took {:.3f} secs'.format(end-start))

        training_loss_epoch.append(training_loss)
        training_end = time.time()

        # validate
        print('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        all_etas = []
        all_meta = []
        start = time.time()
        for graph_data_val, targets_val, meta_val in get_batch(data_gen_val.generator()):#val_iter):
            losses_val, output_vals = val_step(graph_data_val, targets_val)
            targets_val = targets_val.numpy()
            output_vals = output_vals.numpy().squeeze()

            val_loss.append(losses_val.numpy())
            all_targets.append(targets_val)
            all_outputs.append(output_vals)
            all_meta.append(meta_val)

            if not (i)%100:
                end = time.time()
                print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                      format(i, val_loss[-1], np.mean(val_loss)), end='  ')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()

            i += 1 

        end = time.time()
        print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
              format(i, val_loss[-1], np.mean(val_loss)), end='  ')
        print('Took {:.3f} secs'.format(end-start))

        epoch_end = time.time()

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        all_meta = np.concatenate(all_meta)

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
                     outputs=all_outputs,
                     meta=all_meta)
            checkpoint.save(best_ckpt_prefix)
        else:
            print('\nLoss did not decrease from {:.6f}'.format(curr_loss))

        if not (e+1)%5:
            optimizer.learning_rate = optimizer.learning_rate/2
            if optimizer.learning_rate<1e-6:
                optimizer.learning_rate = 1e-6 
                print('\nLearning rate would fall below 1e-6, setting to: {:.5e}'.format(optimizer.learning_rate.value()))
            else:
                print('\nLearning rate decreased to: {:.5e}'.format(optimizer.learning_rate.value()))
