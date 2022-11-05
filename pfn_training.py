import tensorflow as tf
from energyflow.archs import PFN
from training_functions import *
from sklearn.preprocessing import StandardScaler
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# h5_filename = "split_test.hdf5"
# h5_filename = "2M_hcal_update.hdf5"
h5_filename = "2M_hcal_uncompressed_mcPis50GeV+.hdf5"
h5_file = h5.File(h5_filename,'r')

label = "2M_hcal_50GeV-_patience10_10k_batch"  #Replace with your own variation!      
path = "./"+label
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path)

input_dim = h5_file['train_hcal'].shape[-2] #should be 4: Cell E,X,Y,Z, the number of features per particle
learning_rate = 1e-3
dropout_rate = 0.05
batch_size = 1_000
N_Epochs = 50
patience = 5
N_Latent = 128
shuffle_split = True #Turn FALSE for images!
train_shuffle = True #Turn TRUE for images!
Y_scalar = True
loss = 'mse' #'mae' #'swish'

Phi_sizes, F_sizes = (100, 100, N_Latent), (100, 100, 100)
output_act, output_dim = 'linear', 1 #Train to predict error

pfn = PFN(input_dim=input_dim, 
          Phi_sizes=Phi_sizes, 
          F_sizes=F_sizes, 
          output_act=output_act, 
          output_dim=output_dim, 
          loss=loss, 
          latent_dropout=dropout_rate,
          F_dropouts=dropout_rate,
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Tensorflow CallBacks
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay,verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
history_logger=tf.keras.callbacks.CSVLogger(path+"/log.csv", separator=",", append=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint( filepath=path, save_best_only=True)


train_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,'train_hcal','train_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))


val_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,'val_hcal','val_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))

test_generator = tf.data.Dataset.from_generator(
    test_generator(h5_filename,'test_hcal','test_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None])),
    output_types=(tf.float64))

# training_generator.batch(batch_size)
# val_generator.batch(batch_size)
# test_generator.batch(batch_size)

the_fit = pfn.fit(
    train_generator,
    epochs=N_Epochs,
    batch_size=batch_size,
    callbacks=[lr_scheduler, early_stopping,history_logger,model_checkpoint],
    validation_data=val_generator,
    verbose=1
)


pfn.layers
pfn.save("%s/energy_regression.h5"%(path))
mypreds = pfn.predict(test_generator, batch_size=1000)
np.save("%s/predictions.npy"%(path),mypreds)
#FIXME: un-norm the predictions
