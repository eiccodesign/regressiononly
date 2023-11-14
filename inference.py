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
import compress_pickle as pickle

from generator_common import MPGraphDataGenerator
import block as models

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir')
    parser.add_argument('--test_dir')
    parser.add_argument('--num_test_files')
    args = parser.parse_args()
    
    if (args.results_dir is None):
        print("\nNEED TO RUN with --results_dir \n")
        exit()
    elif (args.test_dir is None):
        print("\ NEED TO RUN with --test_dir \n")

    elif (args.num_test_files is None):
        print("\n NEED TO RUN with --num_test_files\n")
    else:
        print("---------------------------------------------")
        print("RESULTS DIR =  ",args.results_dir)
        print(" TEST DIR  ", args.test_dir)
        print("Number of Test Files ", args.num_test_files)
        print("---------------------------------------------")
        print(f"CONFIG = {args.results_dir}/config.yaml")

    config = yaml.safe_load(open(f'{args.results_dir}/config.yaml'))

    save_dir = args.results_dir
    result_dir = args.results_dir

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']
    output_dim=data_config['output_dim']
    if output_dim == 2:
        energy_weight = data_config['energy_weight']
        theta_weight = data_config['theta_weight']
    hadronic_detector = data_config['hadronic_detector']
    data_dir = data_config['data_dir']

    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    num_test_files=int(args.num_test_files)
    test_dir = args.test_dir

    
    print(' here is test directory   ', test_dir)
    print('Number of files used ', num_test_files)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXX')
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    num_features = data_config['num_features']
    k = data_config['k']
    hadronic_detector = data_config['hadronic_detector']
    include_ecal = data_config['include_ecal']
    already_preprocessed = data_config['already_preprocessed']
    print(already_preprocessed, ' already preprocessed -------')
    #already_preprocessed = True
    calc_stats = False
    config['test_dir']=test_dir

    learning_rate = train_config['learning_rate']

    yaml.dump(config, open(save_dir + '/config_inference.yaml', 'w'))


    root_test_files = np.sort(glob.glob(test_dir+'*root'))[:num_test_files]

    #Loads the files from test_dir if specified.
    #Note: if not specified, takes vals from CONFIG
    # if (args.test_dir is not None):
    #     test_dir = args.test_dir

    #     # if not os.path.exists(test_dir+'.root'):
    #     if not any(fname.endswith('.root') for fname in os.listdir(test_dir)):
    #         print(f"\n\nNo ROOT files found in {test_dir}")
    #         print("EXITING\n\n")
    #         exit()

    #     test_files = np.sort(glob.glob(test_dir+'*root'))
    #     print("data_dir = ",data_dir)
    #     print("Test Dir = ",test_dir)
    #     print("Number of Test Files = ",num_test_files)
    #     root_test_files = test_files[:num_test_files]

    # print("\n\n Test Files = ",root_test_files,"\n\n")

    # Get Data
    if preprocess:
        test_output_dir = output_dir + '/test/'

    ### MODEL READ 
    model = models.BlockModel(global_output_size=output_dim, model_config=model_config)
    checkpoint = tf.train.Checkpoint(module=model)

    best_ckpt_prefix = os.path.join(save_dir, '/best_model')
    best_ckpt = tf.train.latest_checkpoint(save_dir)
    last_ckpt_path = result_dir + '/last_saved_model'
    print(output_dim, '-----',)
    
    if os.path.exists(best_ckpt+'.index'):
        print(f'Restoring {best_ckpt}')
        checkpoint.restore(best_ckpt)
    else:
        print("\nCould not load best checkpoint. EXITING\n")
        exit()


    def get_batch(data_iter):
        for graphs, targets, meta in data_iter:
            # data_iter is a triple of lists
            # list of graphs, list of targets, list of meta data with
            # Each entry of the lists has info for one event in the batch
            graphs = convert_to_tuple(graphs)
            # Targets structure: 
            # For 1D: Just a list [ genP0, genP1, genP2, ...]
            # For 2D: list of tuples [ (genP0, gentheta0), (genP1 ,Gentheta1),...]
            # Convert targets to tf.tensor
            # 1D shape (len(targets), ), i.e. [ genP0, genP1, genP2, ...]
            # 2D shape (len(targets), 2), i.e. [ [genP0, gentheta0], [genP1, gentheta1], ...]
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            
            yield graphs, targets
   

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



    #samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
    samp_graph, samp_target = next(get_batch(data_gen_test.generator()))
    data_gen_test.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

    mae_loss = tf.keras.losses.MeanAbsoluteError()

    def loss_fn(targets, predictions):
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

    if output_dim==2:
        provided_shape=[None,None]
    elif output_dim==1:
        provided_shape=[None,]
    
    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=provided_shape, dtype=tf.float32)])
    def val_step(graphs, targets):
            predictions = model(graphs).globals
            # predictions structure:
            # For 1D: predictions (type = tf.Tensor) structure: [ Epred0, Epred1, Epred2, ... ]
            # For 2D: predictions (type = tf.Tensor) structure: [ [Epred0, thetapred0], [Epred1, thetapred1], [Epred2, thetapred2], ...]
            loss = loss_fn(targets, predictions)

            return loss, predictions

    #i = 1
    #test_loss = []
    #all_targets = []
    #all_outputs = []
    #all_targets_scaled = []
    #all_outputs_scaled = []


    means_dict = pickle.load(open(f"{output_dir}/means.p", 'rb'), compression='gzip')
    stdvs_dict = pickle.load(open(f"{output_dir}/stdvs.p", 'rb'), compression='gzip')
    
    def get_pred_2D(data_gen_test, means_dict, stdvs_dict):
        print('Hello Hello')
        i = 1
        test_loss = []
        all_targets = []
        all_outputs = []
        all_targets_scaled = []
        all_outputs_scaled = []
        all_targets_scaled_ene = []
        all_outputs_scaled_ene = []
        all_targets_scaled_theta = []
        all_outputs_scaled_theta = []
        start = time.time()



        for graph_data_test, targets_test in get_batch(data_gen_test.generator()):
            losses_test, output_test = val_step(graph_data_test, targets_test)

            test_loss.append(losses_test.numpy())
            targets_test = targets_test.numpy()
            output_test = output_test.numpy()

            output_test_scaled_ene = 10**(output_test[:,0]*stdvs_dict['genP'] + means_dict['genP'])
            targets_test_scaled_ene = 10**(targets_test[:,0]*stdvs_dict['genP'] + means_dict['genP'])

            output_test_scaled_theta = (output_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])
            targets_test_scaled_theta = (targets_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])

            all_targets.append(targets_test)
            all_outputs.append(output_test)

            all_targets_scaled_ene.append(targets_test_scaled_ene)
            all_outputs_scaled_ene.append(output_test_scaled_ene)

            all_targets_scaled_theta.append(targets_test_scaled_theta)
            all_outputs_scaled_theta.append(output_test_scaled_theta)

            if not (i)%100:
                end = time.time()
                print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
                  format(i, test_loss[-1], np.mean(test_loss)), end='  ')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()

            i += 1

        end = time.time()
        print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
          format(i, test_loss[-1], np.mean(test_loss)), end='  ')
        print('Took {:.3f} secs'.format(end-start))

        epoch_end = time.time()
        all_targets_scaled_theta=np.concatenate(all_targets_scaled_theta)
        all_targets_scaled_ene=np.concatenate(all_targets_scaled_ene)
        all_outputs_scaled_theta=np.concatenate(all_outputs_scaled_theta)
        all_outputs_scaled_ene=np.concatenate(all_outputs_scaled_ene)


        all_targets_scaled=np.vstack((all_targets_scaled_ene, all_targets_scaled_theta)).T
        all_outputs_scaled=np.vstack((all_outputs_scaled_ene, all_outputs_scaled_theta)).T

        return all_targets_scaled, all_outputs_scaled, all_targets, all_outputs


    def get_pred_1D(data_gen_test, means_dict, stdvs_dict):
        print('Hello Hello')
        i = 1
        test_loss = []
        all_targets = []
        all_outputs = []
        all_targets_scaled = []
        all_outputs_scaled = []
        start = time.time()

        for graph_data_test, targets_test in get_batch(data_gen_test.generator()):
            losses_test, output_test = val_step(graph_data_test, targets_test)
            test_loss.append(losses_test.numpy())
            targets_test = targets_test.numpy()
            output_test = output_test.numpy().reshape(-1)

            output_test_scaled = 10**(output_test*stdvs_dict['genP'] + means_dict['genP'])
            targets_test_scaled = 10**(targets_test*stdvs_dict['genP'] + means_dict['genP'])


            all_targets.append(targets_test)
            all_outputs.append(output_test)
            all_targets_scaled.append(targets_test_scaled)
            all_outputs_scaled.append(output_test_scaled)

            if not (i)%100:
                end = time.time()
                print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
                  format(i, test_loss[-1], np.mean(test_loss)), end='  ')
                print('Took {:.3f} secs'.format(end-start))
                start = time.time()

            i += 1

        end = time.time()
        print('Iter: {:03d}, Test_loss_curr: {:.4f}, Test_loss_mean: {:.4f}'. \
          format(i, test_loss[-1], np.mean(test_loss)), end='  ')
        print('Took {:.3f} secs'.format(end-start))

        epoch_end = time.time()
        all_targets_scaled = np.concatenate(all_targets_scaled)
        all_outputs_scaled = np.concatenate(all_outputs_scaled)

        return all_targets_scaled, all_outputs_scaled, all_targets, all_outputs
    if output_dim==1:
        all_targets_scaled, all_outputs_scaled, all_targets, all_outputs=get_pred_1D(data_gen_test, means_dict, stdvs_dict)

    if output_dim==2:
        all_targets_scaled, all_outputs_scaled, all_targets, all_outputs=get_pred_2D(data_gen_test, means_dict, stdvs_dict)

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)    


    print(f"\n Done. Completed {np.shape(all_targets)}\n")
    # print("IGNORE ERROR BELOW")
    # print("       |  |")
    # print("       |  |")
    # print("      \    /")
    # print("       \  /")
    # print("        \/\n")
           
    #np.savez(save_dir+'/predictions_appended_'+'_'.join(save_dir.split('/')[-2].split('_')[-2:])+'.npz', 
    #         targets=all_targets, targets_scaled=all_targets_scaled,
    #         outputs=all_outputs, outputs_scaled=all_outputs_scaled)
    np.savez(save_dir+'/predictions_appended_test.npz',
             targets=all_targets, targets_scaled=all_targets_scaled,
             outputs=all_outputs, outputs_scaled=all_outputs_scaled)
    # np.save(save_dir+'/predictions_standalone.npy', all_outputs)
    # np.save(save_dir+'/targets_standalone.npy', all_targets)

    
