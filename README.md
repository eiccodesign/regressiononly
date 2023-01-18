# Regression Only

The idea of this repo is to learn a regression model conditioned on detector properties.  This can then be combined with a post-hoc optimizer.  The followup to this will be the combination of a regression model and a generative model.

<!-- There are two main types of HDF5 files. The first are `event` HDF5 files. These are simple conversions from the reconstruction ROOT files made in the [generate_data](https://github.com/eiccodesign/generate_data) repository. -->

<!-- The `H5_hitQA` notebook checks `event HDF5 files` -->

<!-- The other type of HDF5 file are `image` HDF5 files. These are heavily processed HDF5 files where an `image` is a convolution of ECal and HCal cell data, with various HCAL segmentations. Each real event yields multiple images (each image contains the same ECAL and HCAL data, but vary the HCAL segmentation). -->
<!-- These files are created by feeding an `event hdf5 file` to the `H5_GetImages` code, found [here](https://github.com/eiccodesign/generate_data/blob/main/to_hdf5/H5_GetImages.cc). -->

The functions folders contains .py files related to plotting and clustering cells. The training folder contains python scripts for running a simple Neural-Network, and a deep-sets model based on the Energy-Flow package.

The pfn training script is located in the training folder, and should be runable from there.

