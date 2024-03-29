# Experiments for Section 3

This subdirectory contains scripts for running the experiments in Section 3.  It also contains data from the experimental results used in the paper.

The data for the experimental results shown in the paper are stored in `data_store/final_results`. 
To load this data, recreate the figures from the paper, and save the figures in `scripts/exp_plots`, run these commands:
```
cd scripts
./make_plots_from_paper.sh
```

A sample workflow for generating data, evaluating metrics, and creating figures like those in the paper is provided below.

First install `interpml` based on instructions in the base directory.  Then run these commands:

```
cd scripts
./batch_data_gen.sh       # saves files to ../data_store
./batch_metric_comp.sh    # saves results to ../data_store/zz_results_dv.csv
./batch_extrap_comp.sh    # saves results to ../data_store/zz_results_extrap.csv
./batch_plot_examples.sh  # saves pdf figures to scripts/exp_plots
```
