#!/bin/bash

##########################################################
# Run this script from the experiments/scripts subdirectory
# Activate an appropriate python env before running
##########################################################

search_dir=../data_store
for entry in "$search_dir"/*npy
do
    # echo "$entry"
    python extrap_comp.py --infile "$entry"
done
