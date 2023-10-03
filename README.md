# Regression Only Repo
### TO DOs
An example plotting notebook, based on functions defined in `plotting.py` that loads the model, does inference (saving to .npy), and plots.

Conversion of train_models.py to a class structure, so we can instantiate the class for standalone inference.

### Introduction
The idea of this repo is to learn a regression model conditioned on detector properties.  This can then be combined with a post-hoc optimizer.  The followup to this will be the combination of a regression model and a generative model.


### Deep Sets Training
The train_models.py is a training script for DeepSets and GNNs (coming soon). It's primary advantage is its ability to train on input data in a permutation invariant way. The GNNs can also directly encode geometric data.

To run `train_models.py` for the first time, go to the `configs` directory, and either edit `default.yaml`, or add your own.
In the configuration file, make sure `data_dir` points to a directory with appropriated data, created using the [generate_data](https://github.com/eiccodesign/generate_data) repository, ideally in the form of several small ROOT files, each with a few thousand events.

Next, edit `already_preprocessed` in the config file to `False` only when running over a dataset for the first time. Afterwards, try running the training:

```python train_models.py```
or 
```python train_models.py --config [config file name]```

One may need to limit `num_procs` and `batch_size` according to what their computer can handle.
