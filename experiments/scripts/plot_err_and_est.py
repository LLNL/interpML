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
    parser.add_option("--infile", dest="infile", type=str, default="../data_store/zz_results_df.csv",
        help="Path to results csv file to read in.")
    parser.add_option( "--case", dest="case", type=int, default=1, 
        help="Case number of plot to make.  Default 1.")
    parser.add_option("--outfile", dest="outfile", type=str, default="temp.pdf",
        help="Name of output file to create.")
    parser.add_option("--quad", action="store_true", default=False,
        help="Only draw a 2x2 table of graphs.  Default False.")
    
    (options, args) = parser.parse_args()

    print("Options selected:")
    print("  file in   =", options.infile)
    print("  case num  =", options.case)
    print("  save file =", options.outfile)

    assert os.getcwd()[-7:] == 'scripts', "Run metric_comp.py from the scripts sub-directory."
    assert exists(options.infile), "There is no file named" + options.infile + "Quitting."


    #################
    # CASES
    ############

    case = options.case
    combine_rbf_and_gp = False
    combine_scaled_and_not = False

    if case == 1:
        xs = 'dims'
        ys = 'spxs'
        colvbl = 'freq'
    elif case == 2:
        xs = 'dims'
        ys = 'gp'
        colvbl = 'freq'
    elif case == 3:
        xs = 'dims'
        ys = 'rbf'
        colvbl = 'freq'
    elif case == 4:
        xs = 'samples'
        ys = 'spxs'
        colvbl = 'freq'
    elif case == 5:
        xs = 'samples'
        ys = 'gp'
        colvbl = 'freq'
    elif case == 6:
        xs = 'samples'
        ys = 'rbf'
        colvbl = 'freq'
    elif case == 7:
        xs = 'samples'
        ys = 'spxs'
        colvbl = 'skew'
    elif case == 8:
        xs = 'samples'
        ys = 'gp'
        colvbl = 'skew'
    elif case == 9:
        xs = 'samples'
        ys = 'rbf'
        colvbl = 'skew'
    elif case == 10:
        xs = 'dims'
        ys = 'dens'
        colvbl = 'freq'
    elif case == 11:
        xs = 'dims'
        ys = 'disc'
        colvbl = 'freq'
    elif case == 12:
        xs = 'dims'
        ys = 'mmxs'
        colvbl = 'freq'
    elif case == 13:
        xs = 'dims'
        ys = 'eOpt'
        colvbl = 'freq'
    elif case == 14:
        xs = 'samples'
        ys = 'mmxs'
        colvbl = 'freq'
    elif case == 15: # same as case 4 but compute est - act diffs
        xs = 'samples'
        ys = 'spxsdiff'
        colvbl = 'freq'
    elif case == 123: # same as case 2 and 3 - but plot both RBF and GP
        xs = 'dims'
        ys = 'rbf'
        colvbl = 'freq'
        combine_rbf_and_gp = True
    elif case == 156: # same as case 5 and 6 - but plot both RBF and GP
        xs = 'samples'
        ys = 'rbf'
        colvbl = 'freq'
        combine_rbf_and_gp = True
    elif case == 107:
        xs = 'samples'
        ys = 'spxs'
        colvbl = 'scale'
        combine_scaled_and_not = True
    elif case == 189:
        xs = 'samples'
        ys = 'rbf'
        colvbl = 'scale'
        combine_scaled_and_not = True
        combine_rbf_and_gp = True
    else:
        print("Case",case,"not recognized, exiting")
        exit()
        
    

    sns.set()
    sns.set_context("talk", font_scale=3)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
    palette_list = sns.color_palette(palette="tab10", n_colors=10)
    # palette_list = sns.color_palette(cc.kbc, n_colors=25)

    mkr_del = 'o'
    mkr_gp = 's'
    mkr_rbf = '^'
    mkr_mlp = 'X'
    
    linesty_err = '-'
    linesty_bd = '--'
 
    clr_del = palette_list[2]
    clr_gp = palette_list[1]
    clr_rbf = palette_list[4]
    clr_mlp = palette_list[6]

    df = pd.read_csv(options.infile, index_col=0, header=0)
    if combine_scaled_and_not:
        print("\n*** Hard coding which file to read ***\n")
        infile_sc = '../data_store/final_results/d5_skew_rescalex_true_FINAL.csv'
        infile_no_sc = '../data_store/final_results/d5_skew_rescalex_false_FINAL.csv'
        df_scaled = pd.read_csv(infile_sc, index_col=0, header=0)
        df_scaled['scaled'] = True
        df_no_sc = pd.read_csv(infile_no_sc, index_col=0, header=0)
        df_no_sc['scaled'] = False
        df = pd.concat([df_scaled,df_no_sc], axis=0, ignore_index=True) 
        df.drop(df[(df.sk == 0.0) & (df.scaled == True)].index , inplace=True)

    if ys in ['spxs', 'gp', 'rbf', 'spxsdiff']:
        numrows = 3
    else:
        numrows = 1
    numcols = 5

    if options.quad:
        if combine_scaled_and_not:
            numrows = 2
            numcols = 3
            print("Dropping skews between 0 and 10...\n\n")
            df.drop(df[(df.sk > 0.0) & (df.sk < 10.0)].index , inplace=True)
        else:
            numrows = 2
            numcols = 2
            print("\n\nDropping frequencies between 0 and 1...")
            df.drop(df[(df.fr > 0.0) & (df.fr < 1.0)].index , inplace=True)
            print("Dropping skews between 0 and 10...\n\n")
            df.drop(df[(df.sk > 0.0) & (df.sk < 10.0)].index , inplace=True)


    # # # add extra col; put legend in last column; will break if nrows > 1
    # f2, ax2 = plt.subplots(nrows=numrows, ncols=numcols+1, sharex=True, sharey=False, 
    #                     squeeze=False, figsize=(5*(numcols)+2.5, 4*numrows)) # figsize=(12,5)

    f2, ax2 = plt.subplots(nrows=numrows, ncols=numcols, sharex=True, sharey='row' if combine_scaled_and_not else False, 
                        squeeze=False, figsize=(5*(numcols)+2.5, 4*numrows)) # figsize=(12,5)


    if case == 15:
        df['diff0.1'] = df['delaunayErrEst0.1'] - df['delaunayIntError0.1']
        df['diff1.0'] = df['delaunayErrEst1.0'] - df['delaunayIntError1.0']
        df['diff2.0'] = df['delaunayErrEst2.0'] - df['delaunayIntError2.0']

    ## sort columns of df before extracting groups - CHECK ORDERING

    if colvbl == 'skew':
        df = df.sort_values(by=['d', 'sk', 'n'], ascending=[True, True, True])
        groups = df.groupby('sk')
    elif colvbl == 'freq':
        df = df.sort_values(by=['d', 'fr', 'n'], ascending=[True, True, True])
        groups = df.groupby('fr')
    elif colvbl == 'scale':
        df = df.sort_values(by=['d', 'fr', 'n'], ascending=[True, True, True])
        groups = df.groupby(['scaled','sk'])
    else:
        print("Colvbl not set! Exiting.")
        exit()

    gpenum = enumerate(groups) # create enumeration object for groups
    col_slot = 0
    for name, group in groups: # if grouping by 'd', name has type int
        if xs == 'dims':
            group = group.groupby('d').mean() # collect over d and average over seeds 
            x_ax = group.index.to_numpy()
        elif xs == 'samples':
            # x_ax = group['n'] # if only one seed
            group = group.groupby('n').mean() # collect over n and average over seeds 
            x_ax = group.index.to_numpy()
        else:
            print("What are x's? Quitting")
            exit()

        ## use linestyle dotted (:) for estimates; linestyle solid (-) for errors

        
        if ys == 'spxs':
            ## radius 0.1
            y_ax1 = group['delaunayErrEst0.1']
            ax2[0][col_slot].plot(x_ax, y_ax1, color=clr_del, marker=mkr_del, linestyle=linesty_bd, markersize=6, label='Delaunay bound')
            y_ax4 = group['delaunayIntError0.1']
            ax2[0][col_slot].plot(x_ax, y_ax4, color=clr_del, marker=mkr_del, linestyle=linesty_err, markersize=6, label='Delaunay error')
            y_ax3 = group['mlpIntErr0.1']
            ax2[0][col_slot].plot(x_ax, y_ax3, color=clr_mlp, marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='MLP error')
            # y_ax9 = group['rfIntErr0.1']
            # ax2[0][col_slot].plot(x_ax, y_ax9, color=palette_list[4], marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='skl rdm forest error')

            ## radius 1
            if not options.quad:
                y_ax1 = group['delaunayErrEst1.0']
                ax2[1][col_slot].plot(x_ax, y_ax1, color=clr_del, marker=mkr_del, linestyle=linesty_bd, markersize=6, label='Deluanay bound')
                y_ax4 = group['delaunayIntError1.0']
                ax2[1][col_slot].plot(x_ax, y_ax4, color=clr_del, marker=mkr_del, linestyle=linesty_err, markersize=6, label='Delaunay error')
                y_ax3 = group['mlpIntErr1.0']
                ax2[1][col_slot].plot(x_ax, y_ax3, color=clr_mlp, marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='MLP error')
                # y_ax9 = group['rfIntErr1.0']
                # ax2[1][col_slot].plot(x_ax, y_ax9, color=palette_list[4], marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='skl rdm forest error')
            
            ## radius 2
            y_ax1 = group['delaunayErrEst2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax1, color=clr_del, marker=mkr_del, linestyle=linesty_bd, markersize=6, label='Deluanay bound')
            y_ax4 = group['delaunayIntError2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax4, color=clr_del, marker=mkr_del, linestyle=linesty_err, markersize=6, label='Delaunay error')
            y_ax3 = group['mlpIntErr2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax3, color=clr_mlp, marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='MLP error')
            # y_ax9 = group['rfIntErr2.0']
            # ax2[numrows-1][col_slot].plot(x_ax, y_ax9, color=palette_list[4], marker=mkr_mlp, linestyle=linesty_err, markersize=6, label='skl rdm forest error')
    
        if ys == 'spxsdiff':
            print("NEEDS UPDATING; exiting")
            exit()
            # y_ax1 = group['diff0.1']
            # ax2[0][col_slot].plot(x_ax, y_ax1, color=palette_list[5], marker='o', linestyle=':', markersize=6, label='avg spx est - DS err')
            # y_ax1 = group['diff1.0']
            # ax2[1][col_slot].plot(x_ax, y_ax1, color=palette_list[5], marker='o', linestyle=':', markersize=6, label='avg spx est - DS err')
            # y_ax1 = group['diff2.0']
            # ax2[2][col_slot].plot(x_ax, y_ax1, color=palette_list[5], marker='o', linestyle=':', markersize=6, label='avg spx est - DS err')


        if ys == 'rbf':
            ## radius 0.1
            y_ax7 = group['rbfErrEst0.1']
            ax2[0][col_slot].plot(x_ax, y_ax7, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_bd, markersize=6, label='RBF bound')
            y_ax8 = group['rbfIntError0.1']
            ax2[0][col_slot].plot(x_ax, y_ax8, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_err, markersize=6, label='RBF error')
            ## radius 1
            if not options.quad:
                y_ax7 = group['rbfErrEst1.0']
                ax2[1][col_slot].plot(x_ax, y_ax7, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_bd, markersize=6, label='RBF bound')
                y_ax8 = group['rbfIntError1.0']
                ax2[1][col_slot].plot(x_ax, y_ax8, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_err, markersize=6, label='RBF error')
            ## radius 2
            y_ax7 = group['rbfErrEst2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax7, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_bd, markersize=6, label='RBF bound')
            y_ax8 = group['rbfIntError2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax8, color=clr_rbf, marker=mkr_rbf, linestyle=linesty_err, markersize=6, label='RBF error')


        if ys == 'gp' or combine_rbf_and_gp:
            ## radius 0.1
            y_ax5 = group['gpErrEst0.1']
            y_ax5 = np.ma.masked_where(y_ax5 == 0, y_ax5)
            ax2[0][col_slot].plot(x_ax, y_ax5, color=clr_gp, marker=mkr_gp, linestyle=linesty_bd, markersize=6, label='GP bound')
            y_ax6 = group['gpIntError0.1']
            ax2[0][col_slot].plot(x_ax, y_ax6, color=clr_gp, marker=mkr_gp, linestyle=linesty_err, markersize=6, label='GP error')
            ## radius 1
            if not options.quad:
                y_ax5 = group['gpErrEst1.0']
                y_ax5 = np.ma.masked_where(y_ax5 == 0, y_ax5)
                ax2[1][col_slot].plot(x_ax, y_ax5, color=clr_gp, marker=mkr_gp, linestyle=linesty_bd, markersize=6, label='GP bound')
                y_ax6 = group['gpIntError1.0']
                ax2[1][col_slot].plot(x_ax, y_ax6, color=clr_gp, marker=mkr_gp, linestyle=linesty_err, markersize=6, label='GP error')
            ## radius 2
            y_ax5 = group['gpErrEst2.0']
            y_ax5 = np.ma.masked_where(y_ax5 == 0, y_ax5)
            ax2[numrows-1][col_slot].plot(x_ax, y_ax5, color=clr_gp, marker=mkr_gp, linestyle=linesty_bd, markersize=6, label='GP bound')
            y_ax6 = group['gpIntError2.0']
            ax2[numrows-1][col_slot].plot(x_ax, y_ax6, color=clr_gp, marker=mkr_gp, linestyle=linesty_err, markersize=6, label='GP error')
            
        if ys in ['dens', 'disc', 'eOpt']:
            y_ax10 = group[ys]
            ax2[0][col_slot].plot(x_ax, y_ax10, color=palette_list[3], marker='*', linestyle=':', markersize=6, label=ys)
        if ys == 'mmxs':
            y_ax11 = group['mxmin']
            ax2[0][col_slot].plot(x_ax, y_ax11, color=palette_list[3], marker='*', linestyle=':', markersize=6, label='maximin')
            y_ax12 = group['mnmax']
            ax2[0][col_slot].plot(x_ax, y_ax12, color=palette_list[4], marker='*', linestyle=':', markersize=6, label='minimax')

        # if case == 3:
        #     ax2[0][col_slot].set_ylim(1e-4,5e-1) # forces y-axis to go from 0 to 1

        # put titles above top row
        if options.quad:
            if colvbl == 'skew':
                ax2[0][0].set_title(r'no skew ($\alpha=0$)', fontsize=18)
                ax2[0][1].set_title(r'high skew ($\alpha=10$)', fontsize=18)
            elif colvbl == 'freq':
                ax2[0][0].set_title(r'low variation ($\omega=0$)', fontsize=18)
                ax2[0][1].set_title(r'high variation ($\omega=1$)', fontsize=18)
            elif colvbl == 'scale':
                if name[0] == False and name[1] == 0.0:
                    ax2[0][col_slot].set_title(r'no skew ($\alpha=0$)', fontsize=18)
                if name[0] == False and name[1] == 10.0:
                    ax2[0][col_slot].set_title(r'high skew ($\alpha=10$), not rescaled', fontsize=18)
                if name[0] == True and name[1] == 10.0:
                    ax2[0][col_slot].set_title(r'high skew ($\alpha=10$), $\{x_i\}$ rescaled', fontsize=18)
            else:
                print("Need col title; exiting")
                exit()
        else:
            if colvbl == 'skew':
                ax2[0][col_slot].set_title('skewness '+str(name), fontsize=18)
            elif colvbl == 'freq':
                ax2[0][col_slot].set_title('frequency '+str(name), fontsize=18)
            elif colvbl == 'scale':
                ax2[0][col_slot].set_title('skewness '+str(name), fontsize=18)
            else:
                print("Need col title; exiting")
                exit()

        if ys in ['spxs', 'gp', 'rbf', 'spxsdiff']:
            if col_slot == 0:
                if options.quad:
                    ax2[0][col_slot].set_ylabel('interpolation', fontsize=18)
                    ax2[numrows-1][col_slot].set_ylabel('extrapolation', fontsize=18)
                else:
                    ax2[0][col_slot].set_ylabel('radius 0.1', fontsize=18)
                    ax2[1][col_slot].set_ylabel('radius 1.0', fontsize=18)
                    ax2[numrows-1][col_slot].set_ylabel('radius 2.0', fontsize=18)


        col_slot += 1

    # set log scale:
    if ys in ['spxs', 'gp', 'rbf', 'dens']:
        for row in range(numrows):
            for col in range(numcols):
                ax2[row][col].set_yscale('log')

    for col in range(numcols):
        if xs == 'samples':
            ax2[numrows-1][col].set_xlabel(r'number of samples ($n$)', fontsize=18)
        elif xs == 'dims':
            ax2[numrows-1][col].set_xlabel(r'dimension ($d$)', fontsize=18)
            ax2[numrows-1][col].xaxis.set_ticks(np.arange(5, 21, 5)) # start, end, stepsize)


    # # put legend in extra slot (use "add extra col" option when defining subplots)
    # for row in range(numrows):
    #     handles, labels = ax2[row][0].get_legend_handles_labels()
    #     ax2[row][numcols].set_frame_on(False)
    #     ax2[row][numcols].get_xaxis().set_visible(False)
    #     ax2[row][numcols].get_yaxis().set_visible(False)
    #     ax2[row][numcols].legend(handles, labels, loc='lower left')

    ## put legend in top right graph
    handles, labels = ax2[0][0].get_legend_handles_labels()
    ax2[0][numcols-1].legend(handles, labels, fontsize=18) # loc='lower left')

    # ax2[0][1].legend(handles, labels, loc='lower left')

    if xs == 'samples':
        plt.xscale('log')

    filename = options.outfile
    plt.savefig(filename,format='pdf', bbox_inches='tight')
    # plt.savefig(filename,format='pdf')

    
    print("\nFinished! saved file",filename,"\n")
    ## uncomment to open the file after generating
    # import subprocess
    # subprocess.call(["open", filename])

