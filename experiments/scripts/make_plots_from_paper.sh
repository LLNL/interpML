#!/bin/bash

##########################################################
# Run this script from the experiments/scripts subdirectory
# Activate an appropriate python env before running
# Figures will be saved in the subdirectory exp_plots
##########################################################

echo 
echo This script must be run from the experiments/scripts subdirectory
echo
mkdir exp_plots
python plot_err_and_est.py --infile ../data_store/final_results/n8192_dim_FINAL.csv --quad --case 1 --outfile exp_plots/dim_del.pdf
python plot_err_and_est.py --infile ../data_store/final_results/n8192_dim_FINAL.csv --quad --case 123 --outfile exp_plots/dim_rbf_and_gp.pdf
python plot_err_and_est.py --infile ../data_store/final_results/d5_freq_FINAL.csv --quad --case 4 --outfile exp_plots/freq_del.pdf
python plot_err_and_est.py --infile ../data_store/final_results/d5_freq_FINAL.csv --quad --case 156 --outfile exp_plots/freq_rbf_and_gp.pdf
python plot_err_and_est.py --infile ../data_store/final_results/d5_skew_rescalex_false_FINAL.csv --quad --case 107 --outfile exp_plots/skew_both_del.pdf
python plot_err_and_est.py --infile ../data_store/final_results/d5_skew_rescalex_false_FINAL.csv --quad --case 189 --outfile exp_plots/skew_both_rbf_and_gp.pdf