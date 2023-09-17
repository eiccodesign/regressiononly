#!/bin/bash
#BSUB -J ecce_deepsets_4D
#BSUB -nnodes 1
#BSUB -W 360
#BSUB -G qcdtq
#BSUB -q pbatch
#BSUB -o outfiles/ECCE_train_deepsets_4D_20230824.out

source ~/.profile.coral

python train_models.py --config configs/lc_lassen_deepsets.yaml
