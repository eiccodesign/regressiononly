sys.path.insert(0, 'functions')
from Clusterer import load_ClusterSum_and_GenP

import tensorflow as tf
from energyflow.utils import data_split #FIXME: switch to sklearn...
import numpy as np


class NN_Regressor:
    def __init__(self,
                 label: str,
                 learning_rate = 1e-5,
                 dropout_rate = 0.05,
                 batch_size = 1000,
                 nEpochs = 400,
                 patience = 20,
                 loss = 'mae'):


        self.label = label
        self.path = "./"+label

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.nEpochs = nEpochs
        self.patience = patience
        self.loss = loss

    def get_callbacks(self):

        lr_decay = tf.keras.callbacks.LearningRateScheduler(step_decay,verbose=0)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.patience)
        self.callbacks = [lr_decay, early_stopping]


    def get_X_Y(self):

        # self.cluster_sum, self.genP = load_ClusterSum_and_GenP(self.label)
        cluster_sum, genP = load_ClusterSum_and_GenP(self.label)

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_split(cluster_sum, genP, val=0.2, test=0.3,shuffle=True)

        return

    def define_model(self):
        self.model = tf.keras.models.Sequential([
                                              tf.keras.layers.Input(shape=[1]),
                                              tf.keras.layers.Dense(64, activation='relu'), 
                                              tf.keras.layers.Dense(64, activation='relu'),
                                              tf.keras.layers.Dense(64, activation='relu'),
                                              tf.keras.layers.Dense(64, activation='relu'),
                                              tf.keras.layers.Dense(1,activation="linear") # output layer.
                                          ])


    def train_model(self):
        self.model.compile(loss=self.loss, optimizer="adam")
        self.fit = self.model.fit(self.x_train, self.y_train, epochs=self.nEpochs,
                              validation_data=(self.x_val,self.y_val),
                              callbacks=self.callbacks,
                              batch_size=self.batch_size)

    def save_results(self):
        self.model.save("%s/energy_regression.h5"%(self.path))
        self.preds = self.model.predict(self.x_test, batch_size=400)    
        np.save("%s/predictions.npy"%(self.path),self.preds)
        np.save("%s/y_test.npy"%(self.path),self.y_test)
        np.save("%s/x_test.npy"%(self.path),self.x_test)
        np.save("%s/loss.npy"%(self.path),self.fit.history["loss"])
        np.save("%s/val_loss.npy"%(self.path),self.fit.history["val_loss"])


    def run_NN_regression(self):
        self.get_X_Y()
        self.get_callbacks()
        self.define_model()
        self.train_model()
        self.save_results()
        print(f"Done! Output saved to {self.path}/")
        
def step_decay(epoch, lr):
    min_rate = 1.01e-7
    N_start = 10
    N_epochs = 5

    return lr

    if epoch >= N_start and lr >= min_rate:
        if (epoch%N_epochs==0):
            return lr * 0.1
    return lr

lr_decay = tf.keras.callbacks.LearningRateScheduler(step_decay,verbose=0)

