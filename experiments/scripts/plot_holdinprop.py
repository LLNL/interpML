import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import colorcet as cc
import os.path
import json

import argparse
from pathlib import Path


sns.set()
sns.set_context("talk", font_scale=3)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
# palette_list = sns.color_palette(palette="tab10", n_colors=20)
# palette_list = sns.color_palette(cc.kbc, n_colors=25)


assert os.getcwd()[-7:] == 'scripts', "Run plot scripts from the scripts sub-directory."
assert os.path.exists('../data_store/zz_results_df.csv'), "Could not find file ../data_store/zz_results_df.csv"

df = pd.read_csv('../data_store/zz_results_df.csv', index_col=0, header=0)


numrows = 2
numcols = 5

# # add extra col; put legend in last column; will break if nrows > 1
f2, ax2 = plt.subplots(nrows=numrows, ncols=numcols+1, sharex=True, sharey=False, 
                       squeeze=False, figsize=(5*(numcols)+2.5, 4*numrows)) # figsize=(12,5)

# xax_d = df['avgss']
# yax_d = df['dsErr']
# color_by = df['d']
# ax2[0][0].scatter(xax_d,yax_d,c=color_by,cmap='Greens')


## sort columns of df before extracting groups - CHECK ORDERING

df = df.sort_values(by=['sk', 'fr', 'n'], ascending=[True, True, True])
groups = df.groupby('d') # assumes no change in sk or fr

col_slot = 0
for name, group in groups: # if grouping by 'd', name has type int
    
    gp_mean = group.groupby('n').mean() # collect over n and average over seeds 
    x_ax = gp_mean.index.to_numpy()
    y_ax105 = gp_mean['holdInProp_5'].to_numpy()
    y_ax110 = gp_mean['holdInProp_10'].to_numpy()

    # print("name",name)
    # print("group",group)
    # print("xax",x_ax)
    # print("yax",y_ax105)
    # exit()
    ax2[0][col_slot].plot(x_ax, y_ax105, marker='o', linestyle='-', markersize=12, label='holdInProp 5%')
    ax2[0][col_slot].set_ylim(0,1.05) # forces y-axis to go from 0 to 1

    ax2[1][col_slot].plot(x_ax, y_ax110, marker='o', linestyle='-', markersize=12, label='holdInProp 10%')
    ax2[1][col_slot].set_ylim(0,1.05) # forces y-axis to go from 0 to 1

    # put titles above top row
    ax2[0][col_slot].set_title('dim '+str(name), fontsize=16)
    # ax2[0][col_slot].set_title('frequency '+str(name), fontsize=16)
    # ax2[0][col_slot].set_title('skewness '+str(name), fontsize=16)

    if col_slot == 0:
        ax2[0][col_slot].set_ylabel(r'interp proportion of 5$\%$ holdout', fontsize=16)
        ax2[1][col_slot].set_ylabel(r'interp proportion of 10$\%$ holdout', fontsize=16)
    col_slot += 1

    # # BELOW: put frequency group on each of three graphs: dsErr, rbfErr, intEst (3 rows, 1 col)
    #
    # yax_0 = group['dsErr']
    # ax2[0][0].plot(x_ax, yax_0, marker='o', linestyle='-', markersize=12, label='fr '+str(name))
    # ax2[0][0].set_ylabel(str(yax_0.name), fontsize=16)
    #
    # yax_1 = group['rbfErr']
    # ax2[1][0].plot(x_ax, yax_1, marker='o', linestyle='-', markersize=12, label='fr '+str(name))
    # ax2[1][0].set_ylabel(str(yax_1.name), fontsize=16)
    #
    # yax_2 = group['intEst']
    # ax2[2][0].plot(x_ax, yax_2, marker='o', linestyle='-', markersize=12, label='fr '+str(name))
    # ax2[2][0].set_ylabel(str(yax_2.name), fontsize=16)

# comment this out for chp or holdInProp:
# for row in range(numrows):
#     for col in range(numcols):
#         ax2[row][col].set_yscale('log')

for col in range(numcols):
    ax2[numrows-1][col].set_xlabel(r'num samples', fontsize=16)

# # put legend in extra slot (use "add extra col" option when defining subplots)

for row in range(numrows):
    handles, labels = ax2[row][0].get_legend_handles_labels()
    ax2[row][numcols].set_frame_on(False)
    ax2[row][numcols].get_xaxis().set_visible(False)
    ax2[row][numcols].get_yaxis().set_visible(False)
    ax2[row][numcols].legend(handles, labels, loc='lower left')


# ax2[0][1].legend(handles, labels, loc='lower left')

plt.xscale('log')

filename = 'zz_holdout_prop_fig.pdf'
plt.savefig(filename,format='pdf', bbox_inches='tight')
# plt.savefig(filename,format='pdf')
import subprocess
subprocess.call(["open", filename])

exit()

plt.yscale('log') # turn on for density plot
handles = labels = []
dims = df['d'].unique()
dims.sort()
skew = df['sk'].unique()
skew.sort()
freq = df['fr'].unique()
freq.sort()

palette_list = sns.color_palette(cc.glasbey, n_colors=len(dims))

for row in range(numrows): # num of rows
    for col in range(numcols): 
        index = row * numcols + col
        # print("col ", col, "skew", skew[col])
        print("Setting up row", row, ", col", col, ", freq", freq[col])
        for i in range(len(dims)-1):
            dim = dims[i+1]
            # tempdf = df.loc[(df['d'] == dim) & (df['sk'] == skew[col])]
            tempdf = df.loc[(df['d'] == dim) & (df['sk'] == 1.0) & (df['fr'] == freq[col])]
            tempdf = tempdf.sort_values('n').reset_index()
            xax_d = tempdf['n']
            # yax_d = tempdf['chp']
            # yax_d = tempdf['dens']
            # yax_d = tempdf['disc']
            # yax_d = tempdf['intEst']
            # yax_d = tempdf['intEst'] - tempdf['dsErr'] 
            if row == 0:
                yax_d = tempdf['dsErr']
                # yax_d = tempdf['spxVol']
            elif row == 1:
                yax_d = tempdf['rbfErr']
            else:
                yax_d = tempdf['gpErr']
            ax1[row][col].plot(xax_d,yax_d, color=palette_list[i], linestyle='-', marker='o', label='dim '+str(dim))

        ax1[row][col].tick_params(axis='x', labelsize=22)
        ax1[row][col].tick_params(axis='y', labelsize=22)
        ax1[row][col].set_xlabel(r'$n$', fontsize=16)
        # ax1[row][col].set_title(r'skew = ' + str(skew[col]), fontsize=16)
        ax1[row][col].set_title(r'freq = ' + str(freq[col]), fontsize=16)

        # ax1[row][col].set_ylim(0,1.05) # forces y-axis to go from 0 to 1
        # ax1[row][col].set_ylim(0,0.4) 
        # ax1[row][col].set_ylabel(r'Normalized density', fontsize=16)

        # # if only one col, use this to put label on y-axis:
        # if col == 0:
        #     ax1[row][col].set_ylabel(r'Interpolation estimate', fontsize=16)
        
        if col == 0:
            # ax1[row][col].set_ylabel(r'(Err est - Del int err) for rad=1', fontsize=16)
            if row == 0:
                ax1[row][col].set_ylabel(r'Delaunay interpolation error', fontsize=16)
                # ax1[row][col].set_ylabel(r'Enclosing Delaunay simplex vol (rad=1)', fontsize=16)
            elif row == 1:
                ax1[row][col].set_ylabel(r'scipy RBF error', fontsize=16)
            else:
                ax1[row][col].set_ylabel(r'scipy Gaussian RBF error', fontsize=16)

        # # get handles for legend:
        handles, labels = ax1[row][col].get_legend_handles_labels()
        # # put legend in each plot:
        # ax1[row][col].legend(loc='lower right', prop={'size': 14})
        
        # # put legend in "last" plot
        # if index == ((numrows-1) * numcols + (numcols-1)):
        #     ax1[row][col].legend(loc='right upper', prop={'size': 10})
    #
#

# # put legend in extra slot (use "add extra col" option when defining subplots)

for row in range(numrows):
    ax1[row][numcols].set_frame_on(False)
    ax1[row][numcols].get_xaxis().set_visible(False)
    ax1[row][numcols].get_yaxis().set_visible(False)
    ax1[row][numcols].legend(handles, labels, loc='lower left')


# f1.suptitle(r'Normalized density vs. number of samples', fontsize=16)
# f1.suptitle(r'Interpolation estimate at origin vs. number of samples ($n$)', fontsize=16)
f1.suptitle(r'Error vs. number of samples ($n$)', fontsize=16)
f1.subplots_adjust(top=0.85) # adds spacing between top row of subplots and title
filename = 'zz_fig.pdf'
plt.savefig(filename,format='pdf', bbox_inches='tight')
# plt.savefig(filename,format='pdf')
import subprocess
subprocess.call(["open", filename])

# plt.show()

#             rlactions = df_rl[df_rl['angle'] == angles[index]][['theta', 'rho']].to_numpy()
#             ax2[i][j].plot(rlactions[:,0],'-o',lw=1.3, c=palette_list[0], label=r'$\theta$ (h parameter)')
#             ax2[i][j].plot(rlactions[:,1],'-o',lw=1.3, c=palette_list[1], label=r'$\rho$ (p parameter)')
#             # ax2[i][j].plot()


