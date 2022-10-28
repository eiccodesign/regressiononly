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

class train_target_generator: #DEPRICATED
    def __init__(self, file, train_dataset, target_dataset):
        self.file = file
        self.train_dataset = train_dataset
        self.target_dataset = target_dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for train, target in zip(hf[self.train_dataset],hf[self.target_dataset]):

                train = np.array(train)
                train = train.T
                # train[:,0] = np.log10(train[:,0]) #Cell E values
                train = (train - train_means) / train_stdevs
                train = np.nan_to_num(train)
                train = train[np.newaxis, :] #PFN expects data as [event# ,particle# ,feature_value_i]

                # if (target[mcE_index][0] > 50): continue
                # target[mcE_index] = np.log10(target[mcE_index])
                target = (target[mcE_index] - target_means[mcE_index,None]) / target_stdevs[mcE_index,None] #can transpose and remove "None"
                target = [target[0]] #particle gun keep first element (sorted for true parent). External brackets for TF iterable

                yield train, target
                # tf.data dataset for training should return a tuple of (inputs, targets)

class test_generator:                                                                                                                                                                    
    def __init__(self, file, train_dataset, target_dataset,batch_size=1000):                                                                                                                 
        self.file = file
        self.train_dataset = train_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.index = 0

    def __call__(self):                                                                                                                                                                      
        with h5py.File(self.file, 'r') as hf:
            
            train_dataset = hf[self.train_dataset] #string to H5 dataset
            target_dataset = hf[self.target_dataset]

            while (self.index < train_dataset.shape[0]):

                start = self.index
                end = start + self.batch_size
                if end > train_dataset.shape[0]: break
                
                train = np.array(train_dataset[start:end])
                train = preprocess_train_data(train)
                                                                                                                           
                target = np.array(target_dataset[start:end])
                target = preprocess_target_data(target)

                self.index += self.batch_size
                yield train 

def preprocess_train_data(data): #Add cell index argument?                                                                                                              
    train_E_index = 0 #careful, should correspond to HDF5 FILE!!!
    data = np.transpose(data, (0,2,1)) #PFN wants [events,particles, features]
    #data[:,E_Index] = np.log10(data[:,train_E_index])                                                                                                                                              
    data = (data - train_means) / train_stdevs                                                                                                                                           
    data = np.nan_to_num(data)                                                                                                                                                               
    return data 

def preprocess_target_data(target): #Add cell index argument?                                                                                                              
    mcE_index = 8
    target = (target[:,mcE_index,:] - target_means[None,mcE_index,None]) / target_stdevs[None,mcE_index,None]
    target = target[:,0] #keep first parent gen particle 
    #target = np.log10(target)
    target = np.nan_to_num(target)

class training_generator:                                                                                                                                                                    
    def __init__(self, file, train_dataset, target_dataset,batch_size=1000):                                                                                                                 
        self.file = file                                                                                                                                                                     
        self.train_dataset = train_dataset                                                                                                                                                   
        self.target_dataset = target_dataset                                                                                                                                                 
        self.batch_size = batch_size                                                                                                                                                         
        self.index = 0                                                                                                                                                                       
                                                                                                                                                                                             
    def __call__(self):                                                                                                                                                                      
        with h5py.File(self.file, 'r') as hf:
            
            train_dataset = hf[self.train_dataset] #string to H5 dataset
            target_dataset = hf[self.target_dataset]

            while (self.index < train_dataset.shape[0]):                                                                                                                                                 
                                                                                                                                                                                             
                start = self.index                                                                                                                                                           
                end = start + self.batch_size                                                                                                                                                     
                if end > train_dataset.shape[0]: break
                                                                                                                                                                                             
                train = np.array(train_dataset[start:end])                                                                          
                train = preprocess_train_data(train)
                                                                                                                                              
                                                                                                                           
                target = np.array(target_dataset[start:end])
                target = preprocess_target_data(target)
                                                                                                                                                                                             
                self.index += self.batch_size                                                                                                                                                     
                yield train, target

def preprocess_train_data(data): #Add cell index argument?                                                                                                              
    train_E_index = 0 #careful, should correspond to HDF5 FILE!!!
    data = np.transpose(data, (0,2,1)) #PFN wants [events,particles, features]
    #data[:,E_Index] = np.log10(data[:,train_E_index])                                                                                                                                              
    data = (data - train_means) / train_stdevs                                                                                                                                           
    data = np.nan_to_num(data)                                                                                                                                                               
    return data 

def preprocess_target_data(target): #Add cell index argument?                                                                                                              
    mcE_index = 8
    target = (target[:,mcE_index,:] - target_means[None,mcE_index,None]) / target_stdevs[None,mcE_index,None]
    target = target[:,0] #keep first parent gen particle 
    #target = np.log10(target)
    target = np.nan_to_num(target)
    return target #should return just the genE of first particles
# class test_generator:
#     def __init__(self, file, dataset):
#         self.file = file
#         self.dataset = dataset

#     def __call__(self):
#         with h5py.File(self.file, 'r') as hf:
#             for data in hf[self.dataset]:
#                 if (data[mcE_index][0] > 50): continue
#                 data = np.array(data)
#                 data = data.T 
#                 # data[:,0] = np.log10(data[:,0])
#                 data = (data - train_means) / train_stdevs
#                 data = np.nan_to_num(data)
#                 data = data[np.newaxis, :] #Needed for PFN (takes 3-D data)
#                 yield data


def scalar_from_generator(tf_dataset, nbatch_stop,E_Index):
    scaler = StandardScaler()
    for data,ibatch in zip(tf_dataset,range(0, nbatch_stop)):

        data = np.array(data)
        data = data.T #cell hit, cell value [EXYZ]. generator class does not return transpose
        data[:,E_Index] = np.log10(data[:,E_Index]) #Log10 on Energy only 
        data = np.nan_to_num(data)
        scaler.partial_fit(data)
        # print("%i / %i \r"%(ibatch,nbatch_stop))
        # print("mean = ",scaler.mean_,"+/-",np.sqrt(scaler.var_))
    return scaler

def lr_decay(epoch, lr):
    min_rate = 1.01e-7
    N_epochs = 10
    N_start = N_epochs - 1

    if epoch > N_start and lr >= min_rate:
        if (epoch%N_epochs==0):
            return lr * 0.1
    return lr

