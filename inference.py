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

import tf2onnx
import onnx

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
    else:
        print("RESULTS DIR = ",args.results_dir)
        print(f"CONFIG = {args.results_dir}/config.yaml")

    config = yaml.safe_load(open(f'{args.results_dir}/config.yaml'))

    save_dir = args.results_dir
    result_dir = args.results_dir

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    data_dir = data_config['data_dir']

    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    
    try:
        num_test_files = data_config['num_test_files']
        test_dir = data_dir
    except:
        if (args.test_dir is None):
            print("\nNEED TO RUN with --test_dir \n")
            exit()
        else:
            test_dir = args.test_dir
            print(f"Test files in = {args.test_dir}")

        if (args.num_test_files is not None):
            num_test_files = int(args.num_test_files)
        else:
            num_test_files = len(glob.glob(test_dir+'/*root'))

    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    num_features = data_config['num_features']
    k = data_config['k']

    z_segmentations = None
    condition_zsections = data_config['condition_zsections']
    print("\n","="*10, "Conditon on Z", condition_zsections, "="*10)

    if condition_zsections:
        z_segmentations = data_config['n_zsections']
        print("\n","="*10, "N Z-sections =", z_segmentations, "="*10)

    hadronic_detector = data_config['hadronic_detector']
    include_ecal = data_config['include_ecal']

    # already_preprocessed = data_config['already_preprocessed']
    # preprocess = True
    already_preprocessed = True
    calc_stats = False

    learning_rate = train_config['learning_rate']

    yaml.dump(config, open(save_dir + '/config_inference.yaml', 'w'))


    root_test_files = np.sort(glob.glob(test_dir+'*root'))[:num_test_files]
    ### Sanity Check
    print(f'\n\nFirst 5 inference root files:\n{root_test_files[:5]}')

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
    model = models.BlockModel(global_output_size=1, model_config=model_config)
    checkpoint = tf.train.Checkpoint(module=model)

    best_ckpt_prefix = os.path.join(save_dir, '/best_model')
    best_ckpt = tf.train.latest_checkpoint(save_dir)
    last_ckpt_path = result_dir + '/last_saved_model'

    if os.path.exists(best_ckpt+'.index'):
        print(f'Restoring {best_ckpt}')
        checkpoint.restore(best_ckpt)
    else:
        print("\nCould not load best checkpoint. EXITING\n")
        exit()



    def get_batch(data_iter):
        for graphs, targets, meta in data_iter:
            graphs = convert_to_tuple(graphs)
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

    #Need to add number of z-segments here. Follow train_models.py Ex.
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
                                         k=k,
                                         n_zsections = z_segmentations,
                                         condition_zsections = condition_zsections)



    #samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
    samp_graph, samp_target = next(get_batch(data_gen_test.generator()))
    print('='*20,"GLOBALS = ", np.shape(samp_graph.globals))
    print('='*20,"Nodes Shape = ", np.shape(samp_graph.nodes))
    print(samp_graph.globals)
    print(samp_graph.nodes)
    # print('='*20,"keys = ",samp_graph)
    data_gen_test.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    input_signature=[graph_spec]

    mae_loss = tf.keras.losses.MeanAbsoluteError()

    def loss_fn(targets, predictions):
            return mae_loss(targets, predictions) 


    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
    def val_step(graphs, targets):
            predictions = model(graphs).globals
            loss = loss_fn(targets, predictions)

            return loss, predictions

    i = 1
    test_loss = []
    all_targets = []
    all_outputs = []
    all_targets_scaled = []
    all_outputs_scaled = []


    means_dict = pickle.load(open(f"{output_dir}/means.p", 'rb'), compression='gzip')
    stdvs_dict = pickle.load(open(f"{output_dir}/stdvs.p", 'rb'), compression='gzip')

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
            
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_targets_scaled = np.concatenate(all_targets_scaled)
    all_outputs_scaled = np.concatenate(all_outputs_scaled)
    # /pscratch/sd/f/fernando/deepsets_models/ECCE_20240110-1855_deepsets_2D
    # tf.save_model(model, result_dir + '/last_saved_model')
    # model.save(result_dir+'/last_saved_model')
    @tf.function(input_signature=input_signature)
    def inference(x):
        return model(x)


    # tf2onnx
    model_proto, _ = tf2onnx.convert.from_function(
        inference,
        input_signature=input_signature, opset=None, custom_ops=None,
        custom_op_handlers=None, custom_rewriter=None,
        inputs_as_nchw=None, extra_opset=None, shape_override=None,
        target=None, large_model=False, output_path="./gnn.onnx")



    tf.saved_model.save(model, result_dir+'/last_saved_model')
    # to_onnx_model = tf.saved_model.load(result_dir+'/last_saved_model')
    # onnx_model, _ = tf2onnx.convert.from_keras(to_onnx_model)
    # onnx_model, _ = tf2onnx.convert.from_keras(to_onnx_model, input_signature)
    # onnx.save(onnx_model, "first_ONNX_model.onnx")

    print(f"\n Done. Completed {np.shape(all_targets)}\n")
    # print("IGNORE ERROR BELOW")
    # print("       |  |")
    # print("       |  |")
    # print("      \    /")
    # print("       \  /")
    # print("        \/\n")

    np.savez(save_dir+'/predictions_appended_'+'_'.join(save_dir.split('/')[-2].split('_')[-2:])+'.npz', 
             targets=all_targets, targets_scaled=all_targets_scaled,
             outputs=all_outputs, outputs_scaled=all_outputs_scaled)
    # np.save(save_dir+'/predictions_standalone.npy', all_outputs)
    # np.save(save_dir+'/targets_standalone.npy', all_targets)
