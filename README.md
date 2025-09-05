# Oveview
This branch is a revamped version of the original branches that uses new classes. This code was used to write the paper "Feasibility study of measuring Λ0→nπ0 using a high-granularity zero-degree calorimeter at the future electron-ion collider"

## How to use
The main parameters to use this code are in `configs/data.yaml` and `configs/training.yaml`.
### data.yaml
This file contains the parameters for the data and preprocessing. Some information about the parameters is given below:
- `OUTPUT_PATH_DIR` is where the preprocessed data will be stored. On the UCR GPU, it is recommended to put the data in `/media/miguel/Elements_2024/AI_data/preprocessed_data/` under some appropriately named directory.
- `TRAINING_FRACTION`/`VALIDATION_FRACTION` the fraction of training to validation data. We use .75 and .25 by default.
- `NUM_PROCESSES` is the number of processes used during preprocessing. This can be set to a high number to preprocess quickly, but should be set back to 2 during training.
- `REGRESSION_VARIABLES` are the variables you want to regress on. The currently supported variables are `momentum`, `theta`, `phi`, `tranverse_momentum`, and `E_minus_pz`.
- `ENERGY_WEIGHT`, `THETA_WEIGHT`, `PHI_WEIGHT` are currently unused and can be ignored.
- `HCAL_NAMES` will contain the names of the HCAL detectors in your data. Available options are "insert", "LFHCAL", "hcal", "zdc_Fe", "zdc_Pb", "muon_detector". Parameters about the HCAL can be set in config_loader.py.
- `ECAL_NAMES` will contain the names of the ECAL detectors in your data. Available options are "zdc_ecal", "ecal_insert", "ecal". Parameters about the ECAL can be set in config_loader.py. If you don't have an ECAL, leave this blank.
- `ROTATE_DATA` should be true if you want to rotate the data to align with the proton axis. This is usually done for data generated with ePIC simulations.
- `USE_THETA_MAX` if you want to use the THETA_MAX variable in config_loader.py
- `USE_ETA_MIN`/`USE_ETA_MAX` if you want to set a min/max value of pseudorapidity. The eta will be calculated in the frame dictated by `ROTATE_DATA`.
- `THETA_UNITS` will be the units of theta if that's a regression variable. Either mrad, rad, deg.
- `USE_ABSOLUTE_VALUE_Z` when this is true, theta will be calculated using `np.arccos(abs(momentum_z)/momentum)`. Otherwise, it'll use `np.arccos(momentum_z/momentum)`.

Classification:
If you want to do classification between two different datasets, set `USE_CLASSIFICATION` to true. You can then set the particle type and paths of the datasets in the options. To not do classification set `USE_CLASSIFICATION` to false and set the paths to your dataset.
Particle type:
Currently there are two special particles: `sigma`, `lambda`, and `rho`. If the particle is set to one of these, a mask will be chosen to pick these particles. Otherwise, the default mask is applied which is `MCParticles.generatorStatus==1`.

### training.yaml
The main parameters you'll want to adjust are `NUM_EPOCHS` and `RESULT_DIR_PATH`. The `RESULT_DIR_PATH` is where the model will be stored. On the UCR GPU, it is recommended to put the model in `/media/miguel/Elements_2024/AI_data/results_and_models/`.

### Running
To preprocess the data and train a model, simply do `python main.py`. After the training is done, run `python inference.py`. 
The model will output a dictionary of predictions and targets.

### Notebooks
After preprocessing, QA plots can be made for the preprocessed data using `notebooks/preprocessed_QA.ipynb`. To see an example of how to use the dictionary output, see `notebooks/testing_new_output.ipynb` or `notebooks/testing_new_output_pt_Eminuspz.ipynb`.

You can check the regression results with the appropriate notebook. For instance, ePIC neutron regression results can be plotted with `notebooks/ePIC_neutron_regression.ipynb` and muography results can be plotted with `notebooks/muography_regression.ipynb`

### Known issues
If you do preprocessing with many processes, you'll likely get an error about too many open files when it transitions to training. To fix this, exit the training (Ctrl+C), change the number of processes to 2, set `CALC_NORMALIZER_STATS` and `PREPROCESS_DATA` to False and run again.
