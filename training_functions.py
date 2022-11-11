import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#Simple function for defining the data generator (interface to data as if it were an infinite stream)
#Mean and standard deviations are obtained from training dataset for performing a standard scalar transformation

input_means = np.load("input_means.npy") #includes log10E
input_stdevs = np.load("input_stdevs.npy") #includes log10E
target_means = np.load("target_means.npy")
target_stdevs = np.load("target_stdevs.npy")

E_Index = 0
mcE_Index = 8 #see to_hdf5
mcTheta_Index = 9

class generator:
    def __init__(self, file, dataset):
        self.file = file
        self.dataset = dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for data in hf[self.dataset]:
                yield data

class test_generator:          
    def __init__(self, file, input_dataset, batch_size=1000,do_norm=True):
        self.file = file
        self.input_dataset = input_dataset
        self.batch_size = batch_size
        self.do_norm = do_norm

    def __call__(self):            
        with h5py.File(self.file, 'r') as hf:
            
            input_dataset = hf[self.input_dataset] #string to H5 dataset

            for start in range(0, input_dataset.shape[0], self.batch_size):

                end = start + self.batch_size
                if end > input_dataset.shape[0]:
                    break
                
                input = np.array(input_dataset[start:end])
                input = preprocess_input_data(input,self.do_norm)

                yield input 


class training_generator:
    def __init__(self, file, input_dataset, target_dataset,batch_size=1000,do_norm=True):
        self.file = file
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.do_norm = do_norm

    def __call__(self):

        with h5py.File(self.file, 'r') as hf:
            input_dataset = hf[self.input_dataset] #string to H5 dataset
            target_dataset = hf[self.target_dataset]

            for start in range(0, input_dataset.shape[0], self.batch_size):
                end = start + self.batch_size                       
                if end > input_dataset.shape[0]: 
                    break
       
                input = np.array(input_dataset[start:end])    
                input = preprocess_input_data(input,self.do_norm)

                target = np.array(target_dataset[start:end])
                target = preprocess_target_data(target,self.do_norm)

                #check for 0's in input (will just regress median)
                if np.all(input[0]==0):
                    continue
                #check for E = NaN
                if np.any(np.isnan(target)):
                    continue
                #check for E = 0.0
                if not np.all(target):
                    continue

                yield input, target

    def quick_scalar(self,n_batches,is_gun=True):

        with h5py.File(self.file, 'r') as hf:

            #INPUT
            input_dataset = hf[self.input_dataset] #string to H5 dataset
            input = input_dataset[:n_batches*self.batch_size]
            input[:,E_Index,:] = np.log10(input[:,E_Index,:])

            input_keys = ["E","X","Y","Z"] # match HDF5 File
            input_means =  np.zeros(len(input_keys))
            input_stdevs = np.zeros(len(input_keys))
            for i in range(len(input_keys)):
                input_means[i]  = np.nanmean(input[:,i,:])
                input_stdevs[i] = np.nanstd (input[:,i,:])

            #TARGET
            target_dataset = hf[self.target_dataset]
            target = target_dataset[:n_batches*self.batch_size]

            target_keys = ["PDG","SimStat","GenStat","mcPX",
                "mcPY","mcPZ","mcMass","mcPT","mcP","mcTheta"]
            target_means = np.zeros(len(target_keys))
            target_stdevs = np.zeros(len(target_keys))
            for i in range(len(target_keys)):
                if (is_gun):
                    target_means[i] = np.nanmean(target[:,i,0])
                    target_stdevs[i] = np.nanstd(target[:,i,0])
                else:
                    target_means[i] = np.nanmean(target[:,i,:])
                    target_stdevs[i] = np.nanstd(target[:,i,:])

            np.save("input_means.npy",input_means)
            np.save("input_stdevs.npy",input_stdevs)
            np.save("target_means.npy",target_means)
            np.save("target_stdevs.npy",target_stdevs)

            print("Input Means = ",input_means)
            print("Input Stdevs = ",input_stdevs)
            print("Target Means = ",target_means)
            print("Target Stdevs = ",target_stdevs)

            return


def preprocess_input_data(data, do_norm=True): #Add cell index argument?            
    data = np.transpose(data, (0,2,1)) #PFN wants [events,particles, features]

    if (do_norm):
        data[:,:,E_Index] = np.log10(data[:,:,E_Index])                
        data = (data - input_means) / input_stdevs             

    data = np.nan_to_num(data)     
    return data 


def preprocess_target_data(target, do_norm=True,is_gun=True): #Add cell index argument?            
    target = np.transpose(target,(0,2,1)) #want [events, gen_part, features]
    target = target[:,:,mcE_Index] # target = target[:,:,mcE_index:] # for Theta too

    if do_norm:
        # target = np.log10(target) #genE is FLAT distribution, shouldn't log10 it.
        target = (target - target_means[mcE_Index]) / target_stdevs[mcE_Index]

    if is_gun:
        target = target[:,0] 

    target = np.nan_to_num(target)
    return target


def scalar_from_generator(tf_dataset, nbatch_stop):
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    for data,ibatch in zip(tf_dataset,range(0, nbatch_stop)):

        input = np.array(data[0]) #event,cell,feature
        input = input.reshape(-1,4)
        print(input[1600:1610,1])
        input[:,E_Index] = np.log10(input[:,E_Index]) #Log10 on Energy only 
        print(input[1600:1610,1])
        # input = np.nan_to_num(input)
        input_scaler.partial_fit(input)
        print(np.shape(input))

        target = np.asarray(data[1]) #event,particle,feature
        target = np.log10(target)
        target = target.reshape(-1,1)
        # target = np.nan_to_num(target)
        target_scaler.partial_fit(target)

        print("%i / %i \r"%(ibatch,nbatch_stop))
        print("input mean = ",input_scaler.mean_,"+/-",np.sqrt(input_scaler.var_))
        print("target mean = ",target_scaler.mean_,"+/-",np.sqrt(target_scaler.var_))

    return input_scaler,target_scaler


def lr_decay(epoch, lr):
    min_rate = 1.01e-7
    N_epochs = 5
    N_start = N_epochs - 1

    if epoch > N_start and lr >= min_rate:
        if (epoch%N_epochs==0):
            return lr * 0.1

    return lr


def pre_training_QA(h5_name, path,n_batches=10,batch_size=1000,do_norm=True):

    print("Doing Final QA before Training")
    cell_vars = ["Energy","Cell X","Cell Y","Cell Z"]
    input_array,target_array,val_input_array,val_target_array,test_array = \
        get_np_from_gen(h5_name,n_batches,batch_size)

    fig,axes=plt.subplots(2,3,figsize=(14,7))
    axes = axes.ravel()
    density = True
    for i in range(input_array.shape[-1]):
        axes[i].hist(np.ravel(input_array[:,:,i][input_array[:,:,i]!=0]),alpha=0.5,label="Training",bins=25,density=density)
        axes[i].hist(np.ravel(val_input_array[:,:,i][val_input_array[:,:,i]!=0]),alpha=0.5,label="Validation",bins=25,density=density)
        axes[i].hist(np.ravel(test_array[:,:,i][test_array[:,:,i]!=0]),alpha=0.5,label="Test",bins=25,density=density)
        axes[i].legend(fontsize=10)
        axes[i].set_title("%s"%(cell_vars[i]),fontsize=20)
    
        #N Cell Hits
        axes[4].hist(np.ravel(np.count_nonzero(input_array[:,:,i],axis=-1)),
                bins=500,alpha=0.2, density=density, label=cell_vars[i])

    axes[4].legend(fontsize=10)
    axes[4].set_title("Number of Cell Hits",fontsize=20)
        
    axes[5].hist(target_array,alpha=0.5,label="Training Truth E",density=density)
    axes[5].hist(val_target_array,alpha=0.5,label="Validation Truth E",density=density)
    axes[5].legend(fontsize=10)
    axes[5].set_title("Validation and Test Truth Energy")
    
    plt.suptitle("Cell Input Data",fontsize=25)
    plt.tight_layout()
        
    if (do_norm):
        plt.savefig("%s/Normalized_Cell_Data.pdf"%(path))
    else:
        plt.savefig("%s/UnNormalized_Cell_Data.pdf"%(path))
    

def get_np_from_gen(h5_filename,n_batches,batch_size=1000,do_norm=True):

    train_generator = tf.data.Dataset.from_generator(
        training_generator(h5_filename,'train_hcal','train_mc',batch_size,do_norm),
        output_shapes=(tf.TensorShape([None,None,None]),[None]),
        output_types=(tf.float64, tf.float64))
 
    val_generator = tf.data.Dataset.from_generator(
        training_generator(h5_filename,'val_hcal','val_mc',batch_size,do_norm),
        output_shapes=(tf.TensorShape([None,None,None]),[None]),
        output_types=(tf.float64, tf.float64))

    testing_generator = tf.data.Dataset.from_generator(
        test_generator(h5_filename,'test_hcal',batch_size,do_norm),
        output_shapes=(tf.TensorShape([None,None,None])),
        output_types=(tf.float64))
    
    base_shape = (1,1861,4)

    #Training Data
    input_array = np.zeros(base_shape)
    target_array = np.zeros(1)
    for data,i in zip(train_generator,range(0,n_batches)):
        input_data = np.asarray(data[0])
        target_data = np.asarray(data[1])
        input_array = np.concatenate((input_array,input_data),axis=0)
        target_array = np.concatenate((target_array,target_data),axis=0)
    
    #Validation Data
    val_input_array = np.zeros(base_shape)
    val_target_array = np.zeros(1)
    for data,i in zip(val_generator,range(0,n_batches)):
        val_input_data = np.asarray(data[0])
        val_target_data = np.asarray(data[1])
        val_input_array = np.concatenate((val_input_array,val_input_data),axis=0)
        val_target_array = np.concatenate((val_target_array,val_target_data),axis=0)

    #Test Data
    test_array = np.zeros(base_shape)
    for test_data,i in zip(testing_generator,range(0,n_batches)):
        test_data = np.asarray(test_data)
        test_array = np.concatenate((test_array,test_data),axis=0)

    return input_array,target_array,val_input_data,val_target_data,test_array




