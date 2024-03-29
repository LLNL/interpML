#!/bin/bash

##########################################################
# Run this script from the experiments/scripts subdirectory
# Activate an appropriate python env before running
# 
##########################################################

echo 
echo This script must be run from the experiments/scripts subdirectory
echo
mkdir exp_plots
python plot_err_and_est.py --infile ../data_store/zz_results_df.csv --quad --case 156 --outfile exp_plots/example_err_plot.pdf
python plot_extrap.py --infile ../data_store/zz_results_extrap.csv --case 5 --outfile exp_plots/example_extrap_plot.pdf
