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
    num_test_files = data_config['num_test_files']
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
    hadronic_detector = data_config['hadronic_detector']
    include_ecal = data_config['include_ecal']
    
    #num_z_layers=data_config['num_z_layers']
    block_type = model_config['block_type']
    print('From the train block model ',num_features, 'output _dim ',  output_dim)
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']

    save_dir = train_config['save_dir'] + '/ECCE_'+time.strftime("%Y%m%d-%H%M_")+block_type+f'_{num_features:d}D'
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    root_files = np.sort(glob.glob(data_dir+'*root'))
    train_start = 0
    train_end = train_start + num_train_files
    val_end = train_end + num_val_files
    test_end = val_end + num_test_files

    root_train_files = root_files[train_start:train_end]
    root_val_files = root_files[train_end:val_end]
    root_test_files = root_files[val_end:test_end]

    train_output_dir = None
    val_output_dir = None
    
    # Get Data
    if preprocess:
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/val/'
        test_output_dir = output_dir + '/test/'
     
    
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
                                          k=k)
                                          
        
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
                                        k=k)
                                        
        
    data_gen_test = MPGraphDataGenerator(file_list=root_test_files,
                                        batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_procs=num_procs,
                                          calc_stats=False,
                                          is_val=True, #decides to save mean and std
                                          preprocess=preprocess,
                                          already_preprocessed=already_preprocessed,
                                          output_dir=test_output_dir,
                                          num_features=num_features,
                                          hadronic_detector=hadronic_detector,
                                          include_ecal=include_ecal,
                                          output_dim=output_dim,
                                          k=k)
                                          
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = models.BlockModel(global_output_size=output_dim, model_config=model_config)

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
        for graphs, targets, meta in data_iter:
            graphs = convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)

            yield graphs, targets
     
    samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)




    mae_loss_energy = tf.keras.losses.MeanAbsoluteError()
    mae_loss_theta = tf.keras.losses.MeanAbsoluteError()
    # Define weights for energy and thet
    weight_energy = 0.8  # Adjust the weight as per your preference
    weight_theta = 0.2  # Adjust the weight as per your preference

    def custom_loss_fn(targets, predictions):
        loss_energy = mae_loss_energy(targets[0], predictions[0]) * weight_energy
        loss_theta = mae_loss_theta(targets[1], predictions[1]) * weight_theta
        total_loss = loss_energy + loss_theta
        return total_loss
    
    mae_loss = tf.keras.losses.MeanAbsoluteError() # Check 
  
    def loss_fn_1D(targets, predictions):
        return mae_loss(targets, predictions)
        #return mae_loss(targets, predictions, sample_weight=[[0.7, 0.3]])
    if output_dim==2:
        provided_shape=[None,None]
        loss_fn=custom_loss_fn
    elif output_dim==1:
        provided_shape=[None,]
        loss_fn=loss_fn_1D

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=provided_shape, dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            predictions = model(graphs).globals
            loss = loss_fn(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=provided_shape, dtype=tf.float32)])
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
        i = 0 # 1 change
        start = time.time()
        for graph_data_tr, targets_tr in get_batch(data_gen_train.generator()):#train_iter):
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
        start = time.time()
        for graph_data_val, targets_val in get_batch(data_gen_val.generator()):#val_iter):
            losses_val, output_vals = val_step(graph_data_val, targets_val)

            targets_val = targets_val.numpy()
            output_vals = output_vals.numpy().squeeze()

            val_loss.append(losses_val.numpy())
            all_targets.append(targets_val)
            all_outputs.append(output_vals)

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

        if not (e+1)%5:
            optimizer.learning_rate = optimizer.learning_rate/2
            if optimizer.learning_rate<1e-6:
                optimizer.learning_rate = 1e-6 
                print('\nLearning rate would fall below 1e-6, setting to: {:.5e}'.format(optimizer.learning_rate.value()))
            else:
                print('\nLearning rate decreased to: {:.5e}'.format(optimizer.learning_rate.value()))

    
    # #Inference over Test Dataset
    print('\nTest Predictions...')
    i = 1
    test_loss = []
    all_targets = []
    all_outputs = []
    all_etas = []
    start = time.time()

    best_ckpt = tf.train.latest_checkpoint(save_dir)
    print(f'Restoring {best_ckpt}')
    checkpoint.restore(best_ckpt)

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

        if not (i)%100:
            end = time.time()
            print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'.format(i, test_loss[-1], np.mean(test_loss)), end='  ')
            print('Took {:.3f} secs'.format(end-start))
            start = time.time()

        i += 1 

    end = time.time()
    print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'.format(i, test_loss[-1], np.mean(test_loss)), end='  ')
    print('Took {:.3f} secs'.format(end-start))

    epoch_end = time.time()

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    np.savez(save_dir+'/test_predictions', 
              targets=all_targets, 
              outputs=all_outputs)
    np.savez(save_dir+'/test_loss', test=test_loss)
    
    
   
