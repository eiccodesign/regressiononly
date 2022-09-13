import h5py
import tensorflow as tf

class generator:
    def __init__(self, file, dataset):
        self.file = file
        self.dataset = dataset

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf[self.dataset]:
                yield im
                #Figure out how to grab chunks, in a way that iteratens through __call__
                #most likeley will use tf.dataset.batch(H5_CHUNK_SIZE)
