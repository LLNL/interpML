import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import colorcet as cc
import os.path
from os.path import exists
from optparse import OptionParser


if __name__ == "__main__":
    """ Driver code """

    usage = "%prog [options]"
    parser = OptionParser(usage)
    parser.add_option("--infile", dest="infile", type=str, default="../data_store/zz_results_extrap.csv",
        help="Path to results csv file to read in.")
    parser.add_option( "--case", dest="case", type=int, default=1, 
        help="Case number of plot to make.  Default 1.")
    parser.add_option("--outfile", dest="outfile", type=str, default="temp.pdf",
        help="Name of output file to create.")
    
    (options, args) = parser.parse_args()

    print("Options selected:")
    print("  file in   =", options.infile)
    print("  case num  =", options.case)
    print("  save file =", options.outfile)

    assert os.getcwd()[-7:] == 'scripts', "Run plot_extrap.py from the scripts sub-directory."
    assert exists(options.infile), "There is no file named" + options.infile + "Quitting."


    #################
    # CASES
    ############

    case = options.case

    if case not in range(1,6):
        print("Case not recognized; exiting")
        exit()

    if case == 1:
        xs = 'samples'
        ys = 'expropcol'
        colvbl = 'dim'
        dims_for_cols = [2,5,8]
    if case == 2:
        xs = 'samples'
        ys = 'hip'
        colvbl = 'dim'
        dims_for_cols = [2,4,8,16]
    if case == 3: # for data gathered using extrap_prop_at_radius; fixed radius --> line charts
        xs = 'radius'
        ys = 'exproprad'
        colvbl = 'dim'
        dims_for_cols = [2,5,8]
    if case == 4: # for data gathered using extrap_with_rad_comp; random radius --> histogram
        xs = 'radius_hist'
        ys = 'exproprad_hist'
        colvbl = 'dim'
        radtype = 'L1'
        dims_for_cols = [2,5,8,11]
    if case == 5: # for data gathered using extrap_with_rad_comp; random radius --> histogram
        xs = 'radius_hist'
        ys = 'exproprad_hist'
        colvbl = 'dim'
        radtype = 'L2'
        dims_for_cols = [2,5,8,11]
        
    

    sns.set()
    sns.set_context("talk", font_scale=3)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    int_color = "b"
    ext_color = "r"

    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
    palette_list = sns.color_palette(palette="tab10", n_colors=10)
    # palette_list = sns.color_palette(cc.kbc, n_colors=25)

    # allrads = []
    # for colname in df.columns:
    #     if 'exProp' in colname:
    #         allrads.append(float(colname[6:]))

    df = pd.read_csv(options.infile, index_col=0, header=0)   
    # print(df)
    
    if xs == 'radius':
        print("==> Melting dataframe to extract rad info.  Note column index is hardcoded.")
        df_melted = pd.melt(df, id_vars=df.columns[:7], value_vars=df.columns[7:], var_name='rad', value_name='exprop')
        def radstr2rad(str_in):
            return float(str_in[6:])
        df_melted['rad'] = df_melted['rad'].apply(radstr2rad)
        df = df_melted

    if xs == 'radius':
        numrows = 1
    elif xs == 'radius_hist':
        numrows = 4
    else:
        numrows = 3
    numcols = len(dims_for_cols)
    
    
    if xs == 'radius_hist':
        f2, ax2 = plt.subplots(nrows=numrows, ncols=numcols, sharex=False, sharey=False, 
                            squeeze=False, figsize=(5*(numcols)+2.5, 4*numrows)) # figsize=(12,5)
    else:
        # # add extra col; put legend in last column; might break if nrows > 1
        f2, ax2 = plt.subplots(nrows=numrows, ncols=numcols+1, sharex=True, sharey=True, 
                            squeeze=False, figsize=(5*(numcols)+2.5, 4*numrows)) # figsize=(12,5)


    ## sort columns of df before extracting groups - CHECK ORDERING


    if colvbl == 'dim':
        df = df.sort_values(by=['d', 'n'], ascending=[True, True])
        groups = df.groupby('d') 
    else:
        print("Colvbl not set! Exiting.")
        exit()

    col_slot = 0
    for name, group in groups: # if grouping by 'd', name has type int
        if int(name) in dims_for_cols:
            if xs == 'samples':
                # x_ax = group['n'] # if only one seed
                # group = group.groupby('n').mean() # collect over n and average over seeds; needs adjustment wrt 'sp'
                # x_ax = group.groupby('n').mean(numeric_only=True).index.to_numpy()
                x_ax = df['n'].unique()
            elif xs == 'radius':
                x_ax = df['rad'].unique()
            elif xs == 'radius_hist':
                pass # will set later
            else:
                print("What are x's? Quitting")
                exit()

            # # use linestyle dotted (:) for estimates; linestyle solid (-) for errors

            if ys == 'expropcol':
                # radius 0.1
                y_ax1 = group['exProp1.0']
                ax2[0][col_slot].plot(x_ax, y_ax1, color=palette_list[0], marker='x', linestyle=':', markersize=6, label='extrap prop')
                # radius 1
                y_ax1 = group['exProp1.5']
                ax2[1][col_slot].plot(x_ax, y_ax1, color=palette_list[0], marker='x', linestyle=':', markersize=6, label='extrap prop')
                # radius 2
                y_ax1 = group['exProp2.0']
                ax2[2][col_slot].plot(x_ax, y_ax1, color=palette_list[0], marker='x', linestyle=':', markersize=6, label='extrap prop')


            if ys == 'exproprad': # name = dim 2, 5, or 8
                print("==> Working on dim",name,"; plotting 1-extrap proportion ***")
                n_groups = group.groupby('n')
                n_name_idx = 0
                for n_name, n_group in n_groups:
                    y_ax = 1-n_group['exprop']
                    ax2[0][col_slot].plot(x_ax,y_ax,color=palette_list[n_name_idx], marker='o', linestyle='-', markersize=10, linewidth=3.0, label=n_name)
                    n_name_idx += 1
            
            if ys == 'exproprad_hist': # name = dim 2, 5, or 8
                print("==> Working on dim",name,"; plotting 1-extrap proportion ***")
                n_groups = group.groupby('n')
                n_name_idx = 0
                for n_name, n_group in n_groups:
                    if radtype == 'L1':
                        x_ax = n_group['L1rad']
                    elif radtype == 'L2':
                        x_ax = n_group['L2rad']
                    else:
                        print("Radtype not set by case. Exiting")
                        exit()
                    y_ax = n_group['exProp']
                    n_bins = 20
                    int_4_hist = x_ax.mask(y_ax>0.5).dropna().reset_index(drop=True)
                    ext_4_hist = x_ax.mask(y_ax<0.5).dropna().reset_index(drop=True)
                    ax2[n_name_idx][col_slot].hist([int_4_hist,ext_4_hist,], range=(0,x_ax.max()), bins=n_bins, stacked=True, color=[int_color,ext_color])
                    if radtype == 'L1':
                        ax2[n_name_idx][col_slot].set_xlim(0,int(name)) # forces x-axis to go from 0 to dim (for L1rad)
                    elif radtype == 'L2':
                        ax2[n_name_idx][col_slot].set_xlim(0,np.sqrt(int(name))) # forces x-axis to go from 0 to dim (for L1rad)
                    if col_slot == 0:
                        ax2[n_name_idx][col_slot].set_ylabel(r'num samples ='+str(n_name), fontsize=16)
                    n_name_idx += 1     

            if ys == 'hip':
                # Sobol sequence
                y_ax1 = group[group['sp'] == 'sob']['holdInProp20']
                y_ax1 = np.pad(y_ax1, (0,len(x_ax)-len(y_ax1)), mode='constant', constant_values=('nan'))
                ax2[0][col_slot].plot(x_ax, y_ax1, color=palette_list[0], marker='o', linestyle='-', markersize=10, linewidth=3.0, label='Sobol')
                # Latin hypercube
                y_ax1 = group[group['sp'] == 'lhc']['holdInProp20']
                y_ax1 = np.pad(y_ax1, (0,len(x_ax)-len(y_ax1)), mode='constant', constant_values=('nan'))
                ax2[1][col_slot].plot(x_ax, y_ax1, color=palette_list[1], marker='o', linestyle='-', markersize=10, linewidth=3.0, label='Latin hypercube')
                # uniform
                y_ax1 = group[group['sp'] == 'uni']['holdInProp20']
                y_ax1 = np.pad(y_ax1, (0,len(x_ax)-len(y_ax1)), mode='constant', constant_values=('nan'))
                ax2[2][col_slot].plot(x_ax, y_ax1, color=palette_list[2], marker='o', linestyle='-', markersize=10, linewidth=3.0, label='Uniform')

            if ys in ['expropcol','exproprad','hip']:
                ax2[0][0].set_ylim(0,1.1) # forces y-axis to go from 0 to 1

            # put titles above top row
            if colvbl == 'dim':
                ax2[0][col_slot].set_title(r'dimension='+str(name), fontsize=16) # +str(name), 
            else:
                print("Need col title; exiting")
                exit()

            if col_slot == 0:
                if ys == 'expropcol':
                    ax2[0][col_slot].set_ylabel('radius 1.0', fontsize=16)
                    ax2[1][col_slot].set_ylabel('radius 1.5', fontsize=16)
                    ax2[2][col_slot].set_ylabel('radius 2.0', fontsize=16)
                if ys == 'hip':
                    ax2[0][col_slot].set_ylabel('Sobol sequence', fontsize=16)
                    ax2[1][col_slot].set_ylabel('Latin hypercube', fontsize=16)
                    ax2[2][col_slot].set_ylabel('Uniform', fontsize=16)
                if ys == 'exproprad':
                    ax2[0][col_slot].set_ylabel('interpolation proportion', fontsize=16)

            col_slot += 1

    for col in range(numcols):
        if xs == 'samples':
            ax2[numrows-1][col].set_xlabel(r'num samples', fontsize=16)
        elif xs == 'radius':
            ax2[numrows-1][col].set_xlabel(r'radius', fontsize=16)
        elif xs == 'radius_hist':
            if radtype == 'L1':
                ax2[numrows-1][col].set_xlabel(r'$L^1$ norm of input', fontsize=16) 
            if radtype == 'L2':
                ax2[numrows-1][col].set_xlabel(r'$L^2$ norm of input', fontsize=16) 
        else:
            print("Not sure what xs is")
            exit()
        
    # # put legend in extra slot (use "add extra col" option when defining subplots)

    if xs == 'radius_hist':
        from matplotlib.patches import Rectangle
        handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [int_color,ext_color]]
        labels= ['interpolation','extrapolation']
        ax2[0][numcols-1].legend(handles, labels, loc='upper left', fontsize=14)
        # plt.legend(handles, labels)
        # handles, labels = ax2[0][0].get_legend_handles_labels()
        # print(handles,labels)
        # exit()
        
    else: # put legend in extra column
        for row in range(numrows):
            handles, labels = ax2[row][0].get_legend_handles_labels()
            ax2[row][numcols].set_frame_on(False)
            ax2[row][numcols].get_xaxis().set_visible(False)
            ax2[row][numcols].get_yaxis().set_visible(False)
            ax2[row][numcols].legend(handles, labels, loc='lower left')

    
    if xs == 'samples':
        plt.xscale('log')

    filename = options.outfile
    plt.savefig(filename,format='pdf', bbox_inches='tight')
    print("\nFinished! saved file",filename,"\n")

    ## uncomment to open the file after generating
    # import subprocess
    # subprocess.call(["open", filename])

