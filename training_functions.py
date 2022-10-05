import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#Simple function for defining the data generator (interface to data as if it were an infinite stream)
#Mean and standard deviations are obtained from training dataset for performing a standard scalar transformation
#FIXME: should do log10(Energy) before standard scalar...

train_means = np.asarray([5.03507405e+00,  3.18427509e-01, -2.18519480e+00,  4.32908556e+03])
train_stdevs = np.asarray([12.82926836, 1149.09226927, 1164.31439935,  292.47299661])

target_means =  np.asarray([2.11000000e+02, 2.16756598e+07, 1.00000000e+00, -3.86635313e-02, -2.54282110e-02, 4.73076933e+01, 1.39570177e-01, 1.49288059e+01, 5.00046094e+01, 1.75126818e+01])

target_stdevs = np.asarray([0.00000000e+00, 2.50778893e+07, 0.00000000e+00, 1.31630895e+01, 1.31124268e+01, 2.73733162e+01, 0.00000000e+00, 1.10605406e+01, 2.88454524e+01, 7.22166879e+00])

mcE_index = 8; #see to_hdf5

class generator:
    def __init__(self, file, dataset):
        self.file = file
        self.dataset = dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for data in hf[self.dataset]:
                yield data

                # A tf.data dataset. Should return a tuple of either (inputs, targets)

class train_target_generator:
    def __init__(self, file, train_dataset, target_dataset):
        self.file = file
        self.train_dataset = train_dataset
        self.target_dataset = target_dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for train, target in zip(hf[self.train_dataset],hf[self.target_dataset]):

                train = (train - train_means[:,None]) / train_stdevs[:,None]
                target = (target[mcE_index] - target_means[mcE_index,None]) / target_stdevs[mcE_index,None]
                # print(target[0])
                # print(type(target))

                yield (train, target[0])

                # A tf.data dataset. Should return a tuple of either (inputs, targets)

def scalar_from_generator(tf_dataset, nvars, nbatch_stop):
    scaler = StandardScaler()
    for data,ibatch in zip(tf_dataset.batch(1000),range(0, nbatch_stop)):
        scaler.partial_fit(data.numpy().transpose(0,2,1).reshape(-1,nvars))
        print("mean = ",scaler.mean_,"+/-",np.sqrt(scaler.var_))


def lr_decay(epoch, lr):
    min_rate = 1.01e-7
    N_epochs = 40
    N_start = N_epochs - 1

    if epoch > N_start and lr >= min_rate:
        if (epoch%N_epochs==0):
            return lr * 0.1
    return lr

# def partial_scalar_fit():
# for data in tf_dataset.batch(1000):   
#     scaler.partial_fit(data.numpy().transpose(0,2,1).reshape(-1,4))
# print("mean = ",scaler.mean_,"+/-",np.sqrt(scaler.var_))
