import os
import time
from graph_nets.graphs      import GraphsTuple
from graph_nets             import utils_tf
from typing                 import Generator, Tuple
from tqdm                   import tqdm

import numpy                as np
import tensorflow           as tf
import logging

from config_loader          import ConfigLoader
from exceptions             import ModelException
from data_generator         import DataGenerator



class Model:

    """
    This class handles all operations relevant to the model. The config
    class must be passed in to instantiate a class. Then, the optimizer,
    loss function, and model must be set through the set functions
    available below. Then the train function can be called to train
    the model. 
    """

    def __init__(
            self, 
            config_loader: ConfigLoader, 
            model: tf.keras.models,
    ):
        
        self.config = config_loader
        self.model = model
        "Setup checkpointing"
        self.checkpoint = tf.train.Checkpoint(module=self.model)

        "Automatically determine model parameters"
        self.set_regression_loss_fn()
        self.set_optimizer()

        "Initialize logger for error warnings, and info"
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)


    def train_model(self, val_data: DataGenerator, train_data: DataGenerator):
        config = self.config

        self._wrapped_train_step = self._create_wrapped_train_step(train_data)
        self._wrapped_val_step = self._create_wrapped_val_step(train_data)
        
        training_loss_epoch = []
        val_loss_epoch = []
        curr_loss = 1e5
        checkpoint = tf.train.Checkpoint(module=self.model)
        best_ckpt_prefix = os.path.join(config.RESULT_DIR_PATH, 'best_model')
        best_ckpt = tf.train.latest_checkpoint(config.RESULT_DIR_PATH)
        last_ckpt_path = config.RESULT_DIR_PATH + '/last_saved_model'

        if not os.path.exists(config.RESULT_DIR_PATH):
            os.makedirs(config.RESULT_DIR_PATH)

        #Main Epoch Loop
        for epoch in range(config.NUM_EPOCHS):
            print(f'\nStarting epoch: {epoch}')
            epoch_start = time.time()
            training_loss = []
            val_loss = []

            # Train
            print('\nStarting Training')
            train_gen = self._get_batch(train_data.generator())
            with tqdm(train_gen, desc="Training") as pbar:
                for i, (graph_data_tr, targets_tr, _) in enumerate(pbar):
                    start = time.time()
                    losses_tr = self._train_step(graph_data_tr, targets_tr)
                    training_loss.append(losses_tr.numpy())
                    end = time.time()
                    pbar.set_postfix({
                        'Tr_loss_curr': f'{training_loss[-1]:.4f}', 
                        'Tr_loss_mean': f'{np.mean(training_loss):.4f}', 
                        'Took': f'{end-start:.3f} secs'
                    })
                training_loss_epoch.append(training_loss)
                training_end = time.time()

            # validate
            print('\nStarting Validation')
            all_targets = []
            all_outputs = []
            all_etas = []
            all_meta = []

            val_gen = self._get_batch(val_data.generator())
            with tqdm(val_gen, desc="Validation") as pbar:
                for i, (graph_data_val, targets_val, meta_val) in enumerate(pbar):
                    start = time.time()
                    losses_val, output_vals = self._val_step(graph_data_val, targets_val)
                    targets_val = targets_val.numpy()
                    output_vals = output_vals.numpy().squeeze()
                    val_loss.append(losses_val.numpy())
                    all_targets.append(targets_val)
                    all_outputs.append(output_vals)
                    all_meta.append(meta_val)
                    end = time.time()
                    pbar.set_postfix({
                        'Val_loss_curr': f'{val_loss[-1]:.4f}', 
                        'Val_loss_mean': f'{np.mean(val_loss):.4f}', 
                        'Took': f'{end-start:.3f} secs'
                    })

            epoch_end = time.time()

            all_targets = np.concatenate(all_targets)
            all_outputs = np.concatenate(all_outputs)
            all_meta = np.concatenate(all_meta)
            val_loss_epoch.append(val_loss)

            np.savez(
                config.RESULT_DIR_PATH+'/losses', 
                training=training_loss_epoch, 
                validation=val_loss_epoch
            )
            checkpoint.write(last_ckpt_path)

            val_mins = int((epoch_end - training_end)/60)
            val_secs = int((epoch_end - training_end)%60)
            training_mins = int((training_end - epoch_start)/60)
            training_secs = int((training_end - epoch_start)%60)

            print('\nEpoch {} ended\nTraining: {:2d}:{:02d}\nValidation: {:2d}:{:02d}'. \
                format(epoch, training_mins, training_secs, val_mins, val_secs))

            if np.mean(val_loss)<curr_loss:
                print('Loss decreased from {:.6f} to {:.6f}'.format(curr_loss, np.mean(val_loss)))
                print('Checkpointing and saving predictions to:\n{}'.format(config.RESULT_DIR_PATH))
                curr_loss = np.mean(val_loss)
                np.savez(
                    config.RESULT_DIR_PATH+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs,
                    meta=all_meta
                )
                checkpoint.save(best_ckpt_prefix)
            else:
                print('Loss did not decrease from {:.6f}'.format(curr_loss))
            self.optimizer.learning_rate = self.optimizer.learning_rate / 2
            if self.optimizer.learning_rate<1e-6:
                self.optimizer.learning_rate = 1e-6 
                print('Learning rate would fall below 1e-6, setting to: {:.5e}'.format(self.optimizer.learning_rate.value()))
            else:
                print('Learning rate decreased to: {:.5e}'.format(self.optimizer.learning_rate.value()))


    """
    The below functions are the private helper functions responsible for
    training the model. 
    """

    def _get_input_signature(self, data_generator: DataGenerator) -> list:
        samp_graph, _, _ = next(self._get_batch(data_generator.generator()))
        data_generator.kill_processes()
        graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

        if self.config.OUTPUT_DIMENSIONS == 1:
            self.provided_shape = [None,]
        elif self.config.OUTPUT_DIMENSIONS == 2 or self.config.OUTPUT_DIMENSIONS == 3:
            self.provided_shape = [None, None]
        else:
            raise ModelException(
                f"Unsupported OUTPUT_DIMENSION: {self.config.OUTPUT_DIMENSION}"
            )
        return [
            graph_spec,
            tf.TensorSpec(
                shape=self.provided_shape,
                dtype=tf.float32
            )
        ]


    def _create_wrapped_train_step(self, data_generator: DataGenerator):
        @tf.function(input_signature=self._get_input_signature(data_generator))
        def _wrapped_train_step(graphs, targets):
            model = self.model
            with tf.GradientTape() as tape:
                predictions = model(graphs).globals
                loss = self.regression_loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        return _wrapped_train_step


    def _train_step(self, graphs, targets) -> float:
        return self._wrapped_train_step(graphs, targets)


    def _create_wrapped_val_step(self, data_generator: DataGenerator):
        @tf.function(input_signature=self._get_input_signature(data_generator))
        def _wrapped_val_step(graphs, targets):
            model = self.model
            predictions = model(graphs).globals
            loss = self.regression_loss_fn(targets, predictions)
            return loss, predictions
        return _wrapped_val_step



    def _val_step(self, graphs, targets) -> float:
        return self._wrapped_val_step(graphs, targets)
    

    def _get_batch(self, data_iter) -> Generator[Tuple, Tuple, any]:
        """     
        data_iter is a tuple of lists
        list of graphs, list of targets, list of meta data with
        Each entry of the lists has info for one event in the batch
        
         targets structure: 
            For 1D: Just a list [ genP0, genP1, genP2, ...]
            For 2D: list of tuples [ (genP0, gentheta0),...]
            Convert targets to tf.tensor
            1D shape (len(targets), ), i.e. [ genP0, genP1, genP2, ...]
            2D shape (len(targets), 2), i.e. [ [genP0, gentheta0],  ...]
        """

        for graphs, targets, meta in data_iter:

            graphs = self._convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            yield graphs, targets, meta


    def _convert_to_tuple(self, graphs: list) -> GraphsTuple:
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
            n_node.append([len(graph['nodes'])])

            if graph['senders'] is not None:
                senders.append(graph['senders'] + offset)
            if graph['receivers'] is not None:
                receivers.append(graph['receivers'] + offset)
            if graph['edges'] is not None:
                edges.append(graph['edges'])
                n_edge.append([len(graph['edges'])])
            else:
                n_edge.append([0])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes), dtype=tf.float32)
        globals = tf.convert_to_tensor(np.concatenate(globals), dtype=tf.float32)
        n_node = tf.convert_to_tensor(np.concatenate(n_node), dtype=tf.int64)
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge), dtype=tf.int64)

        if senders:
            senders = tf.convert_to_tensor(np.concatenate(senders), dtype=tf.int32)
        else:
            senders = tf.convert_to_tensor([], dtype=tf.int32)

        if receivers:
            receivers = tf.convert_to_tensor(np.concatenate(receivers), dtype=tf.int32)
        else:
            receivers = tf.convert_to_tensor([], dtype=tf.int32)

        if edges:
            edges = tf.convert_to_tensor(np.concatenate(edges), dtype=tf.float32)
        else:
            edges = tf.convert_to_tensor([], dtype=tf.float32)
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
    

    """
    These functions can be called to set the loss function and optimizer.
    If not called the default optimizer and loss function will be used 
    (MAE and Adam). 
    """

    def set_optimizer(self):
        config = self.config

        if config.OPTIMIZER.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE
            )
        elif config.OPTIMIZER.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=config.LEARNING_RATE
            )
        elif config.OPTIMIZER.lower() == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=config.LEARNING_RATE 
            )
        elif config.OPTIMIZER.lower() == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=config.LEARNING_RATE
            )
        elif config.OPTIMIZER.lower() == 'adamax':
            self.optimizer = tf.keras.optimizers.Adamax(
                learning_rate=config.LEARNING_RATE
            )
        elif config.OPTIMIZER.lower() == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=config.LEARNING_RATE 
            )
        else:
            raise ModelException(
                f"Unknown optimizer: {config.OPTIMIZER}"
            )


    def set_regression_loss_fn(self ):
        config = self.config

        if config.LOSS_FUNCTION.lower() == 'mae':
            self.regression_loss_fn = (
                tf.keras.losses.MeanAbsoluteError()
            )
        elif config.LOSS_FUNCTION.lower() == 'mae':
            self.regression_loss_fn = (
                tf.keras.losses.MeanSquaredError()
            )
        elif config.LOSS_FUNCTION.lower() == 'huber':
            self.regression_loss_fn = (
                tf.keras.losses.Huber()
            )
        elif config.LOSS_FUNCTION.lower() == 'mape':
            self.regression_loss_fn = (
                tf.keras.losses.MeanAbsolutePercentageError()
            )
        elif config.LOSS_FUNCTION.lower() == 'msle':
            self.regression_loss_fn = (
                tf.keras.losses.MeanSquaredLogarithmicError()
            )
        elif config.LOSS_FUNCTION.lower() == 'log_cosh':
            self.regression_loss_fn = (
                tf.keras.losses.LogCosh()
            )
        else:
            raise ModelException(
                f"Unknown loss function: {config.LOSS_FUNCTION}"
            )


    def get_pred_3D(self, data_generator: DataGenerator, means_dict: dict, stdvs_dict: dict):
        self._wrapped_val_step = self._create_wrapped_val_step(data_generator)

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
        all_targets_scaled_phi = []
        all_outputs_scaled_phi = []
        all_meta = []
        start = time.time()


        for graph_data_test, targets_test, meta_test in self._get_batch(data_generator.generator()):
            losses_test, output_test = self._val_step(graph_data_test, targets_test)

            test_loss.append(losses_test.numpy())
            targets_test = targets_test.numpy()
            output_test = output_test.numpy()

            output_test_scaled_ene = 10**(output_test[:,0]*stdvs_dict['momentum'] + means_dict['momentum'])
            targets_test_scaled_ene = 10**(targets_test[:,0]*stdvs_dict['momentum'] + means_dict['momentum'])

            output_test_scaled_theta = (output_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])
            targets_test_scaled_theta = (targets_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])

            output_test_scaled_phi = (output_test[:,2]*stdvs_dict['phi'] + means_dict['phi'])
            targets_test_scaled_phi = (targets_test[:,2]*stdvs_dict['phi'] + means_dict['phi'])

            all_targets.append(targets_test)
            all_outputs.append(output_test)
            all_meta.append(meta_test)

            all_targets_scaled_ene.append(targets_test_scaled_ene)
            all_outputs_scaled_ene.append(output_test_scaled_ene)

            all_targets_scaled_theta.append(targets_test_scaled_theta)
            all_outputs_scaled_theta.append(output_test_scaled_theta)

            all_targets_scaled_phi.append(targets_test_scaled_phi)
            all_outputs_scaled_phi.append(output_test_scaled_phi)

        all_targets_scaled_theta=np.concatenate(all_targets_scaled_theta)
        all_targets_scaled_ene=np.concatenate(all_targets_scaled_ene)
        all_outputs_scaled_theta=np.concatenate(all_outputs_scaled_theta)
        all_outputs_scaled_ene=np.concatenate(all_outputs_scaled_ene)
        all_targets_scaled=np.vstack((all_targets_scaled_ene, all_targets_scaled_theta)).T
        all_outputs_scaled=np.vstack((all_outputs_scaled_ene, all_outputs_scaled_theta)).T
        all_meta = np.concatenate(all_meta)

        return all_targets_scaled, all_outputs_scaled, all_targets, all_outputs, all_meta


    def get_pred_2D(self, data_generator: DataGenerator, means_dict: dict, stdvs_dict: dict):
        self._wrapped_val_step = self._create_wrapped_val_step(data_generator)

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
        all_meta = []
        start = time.time()


        for graph_data_test, targets_test, meta_test in self._get_batch(data_generator.generator()):
            losses_test, output_test = self._val_step(graph_data_test, targets_test)

            test_loss.append(losses_test.numpy())
            targets_test = targets_test.numpy()
            output_test = output_test.numpy()

            output_test_scaled_ene = 10**(output_test[:,0]*stdvs_dict['momentum'] + means_dict['momentum'])
            targets_test_scaled_ene = 10**(targets_test[:,0]*stdvs_dict['momentum'] + means_dict['momentum'])

            output_test_scaled_theta = (output_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])
            targets_test_scaled_theta = (targets_test[:,1]*stdvs_dict['theta'] + means_dict['theta'])

            all_targets.append(targets_test)
            all_outputs.append(output_test)
            all_meta.append(meta_test)

            all_targets_scaled_ene.append(targets_test_scaled_ene)
            all_outputs_scaled_ene.append(output_test_scaled_ene)

            all_targets_scaled_theta.append(targets_test_scaled_theta)
            all_outputs_scaled_theta.append(output_test_scaled_theta)

        all_targets_scaled_theta=np.concatenate(all_targets_scaled_theta)
        all_targets_scaled_ene=np.concatenate(all_targets_scaled_ene)
        all_outputs_scaled_theta=np.concatenate(all_outputs_scaled_theta)
        all_outputs_scaled_ene=np.concatenate(all_outputs_scaled_ene)
        all_targets_scaled=np.vstack((all_targets_scaled_ene, all_targets_scaled_theta)).T
        all_outputs_scaled=np.vstack((all_outputs_scaled_ene, all_outputs_scaled_theta)).T
        all_meta = np.concatenate(all_meta)

        return all_targets_scaled, all_outputs_scaled, all_targets, all_outputs, all_meta


    def get_pred_1D(self, data_generator: DataGenerator, means_dict: dict, stdvs_dict: dict):
        self._wrapped_val_step = self._create_wrapped_val_step(data_generator)

        i = 1
        test_loss = []
        all_targets = []
        all_outputs = []
        all_targets_scaled = []
        all_outputs_scaled = []
        all_meta = []
        start = time.time()

        for graph_data_test, targets_test, meta_test in self._get_batch(data_generator.generator()):
            losses_test, output_test = self._val_step(graph_data_test, targets_test)
            test_loss.append(losses_test.numpy())
            targets_test = targets_test.numpy()
            output_test = output_test.numpy().reshape(-1)

            output_test_scaled = 10**(output_test*stdvs_dict['momentum'] + means_dict['momentum'])
            targets_test_scaled = 10**(targets_test*stdvs_dict['momentum'] + means_dict['momentum'])


            all_targets.append(targets_test)
            all_outputs.append(output_test)
            all_meta.append(meta_test)

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
        all_meta = np.concatenate(all_meta)

        return all_targets_scaled, all_outputs_scaled, all_targets, all_outputs, all_meta
    
