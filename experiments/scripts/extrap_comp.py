# Data quality repo
#   Extrapolation computation routines
#   Saves to file zz_results_extrap.csv

import numpy as np
from optparse import OptionParser
import os
import pandas as pd
from data_metrics import *
from data_gen import *


if __name__ == "__main__":
    """ Driver code """

    usage = "%prog [options]"
    parser = OptionParser(usage)
    parser.add_option("--infile", dest="infile", type=str, default=None,
        help="Path to .npy file to read in.")
    
    (options, args) = parser.parse_args()

    assert os.getcwd()[-7:] == 'scripts', "Run extrap_comp.py from the scripts sub-directory."

    print("Options selected:")
    print("  file in   =", options.infile)
    print()

    assert os.path.exists(options.infile), "There is no file named" + options.infile + "  Quitting."
    
    f = options.infile

    spi = f.find('_sp_')
    di  = f.find('_d_')
    fri = f.find('_fr_')
    ski = f.find('_sk_')
    sdi = f.find('_sd_')
    ni  = f.find('_n_')
    si  = f.find('_seed_')
    ei  = f.find('.npy')

    if spi == -1: # legacy file; used Sobol sampling
        sp = 'sob'
    else:
        sp = str(f[spi+4:di])
    d    = int(f[di+3:fri])
    fr   = float(f[fri+4:ski])
    sk   = float(f[ski+4:sdi])
    sd   = float(f[sdi+4:ni])
    n    = int(f[ni+3:si])
    seed = int(f[si+6:ei])

    indata = np.load(options.infile)

    assert indata.shape[1] == d + 1, "Number of data columns doesn't match filename."
    assert indata.shape[0] == n, "Number of data rows doesn't match filename."

    X = indata[:,:-1] # numpy array of inputs
    Y = indata[:,-1]  # numpy array of outputs 

    # use seed from infile to create rng
    rng = np.random.default_rng(seed) 

    ################################################
    ##  Check if already ran metrics on this dataset
    ################################################

    result_row_index = None
    results_filename = os.path.dirname(options.infile) + '/zz_results_extrap.csv' # results file in same directory as data
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename, index_col=0)
        if 'sp' not in results_df.columns: # legacy file; add column to indicate Sobol sampling
            results_df['sp'] = 'sob'
        matching_rows = results_df.index[
                (results_df['sp'] == sp) &
                (results_df['d'] == d) & 
                (results_df['fr'] == fr) & 
                (results_df['sk'] == sk) & 
                (results_df['sd'] == sd) & 
                (results_df['n'] == n) & 
                (results_df['seed'] == seed)]
        if len(matching_rows) == 1:
            result_row_index = matching_rows[0]
            # print("Found matching row - exiting")
            # exit()
        elif len(matching_rows) > 1:
            print("==> WARNING: duplicate rows exist for same data sample; exiting")
            exit()
            # print("==> Allowing mutliple matching rows for extrap_with_rad_comp")
        else:
            pass # a row for this datasset has not been created in the results file

    all_ex_dicts = [] # one list used across all extrap subroutines


    ##############################################
    #  Extrap proportion for random draws in domain
    #  --> Draws M points in [-1,1]^d (default M=100)
    #  --> For each point, saves L^1 and L^2 radius from origin
    #       and whether it is interpolation or extrapolation
    ##############################################
    
    def extrap_with_rad_comp(test_pt):


        # # create test points at specified radius
        # for i in range(num_pts_to_test):
        #     dir_vec = rng.normal(size=(X.shape[1]))
        #     norm_vec = dir_vec/np.linalg.norm(dir_vec)
        #     eval_pt = eval_rad * norm_vec
        #     test_pts[i] = eval_pt
        # #

        extrap_dict = {} # empty dictionary to hold result

        interior_prop = convexHullPercent(X, test_pt) # returns 1 if interp; 0 if extrap
        extrap_prop = 1 - interior_prop
        L2norm = np.linalg.norm(test_pt,ord=2,axis=1)
        L1norm = np.linalg.norm(test_pt,ord=1,axis=1)

        extrap_dict['exProp'] = extrap_prop
        extrap_dict['L2rad'] = L2norm
        extrap_dict['L1rad'] = L1norm
        # print(extrap_dict)

        return extrap_dict
    
    num_pts_to_test = 2**10

    # # uniform sampling:
    test_pts = rng.uniform(size=(num_pts_to_test, X.shape[1])) # draw uniformly

    # # sobol sampling
    # from scipy.stats import qmc
    # sampler = qmc.Sobol(X.shape[1], scramble=True, seed=rng)
    # test_pts = sampler.random(num_pts_to_test)

    # Transform test_pts from [0,1]^d to [-1,1]^d
    test_pts = (2 * test_pts - 1)
    # tpmax = np.linalg.norm(test_pts,ord=1,axis=1).max()
    # print("test pt max =", tpmax)

    test_pts_list = list(enumerate(test_pts))
    for idx, pt in test_pts_list:
        pt = pt.reshape((1,-1))
        tpl = np.linalg.norm(pt,ord=1,axis=1)
        # if tpl > 0.9*tpmax:
        #     print("testing pt of length",tpl)
        extrap_dict = extrap_with_rad_comp(pt)
        extrap_dict['ptidx'] = int(idx)
        all_ex_dicts.append(extrap_dict)

    result_row_index = None # ensures all data will be appended to existing df
    

    ##############################################
    ##  Create new row for results, if needed
    ##############################################

    if result_row_index is None:
        ################################################
        ##  Prep data frame of results
        ################################################

        new_results_raw = {
                    "sp": [sp],
                    "d" : [d],
                    "fr": [fr],
                    "sk": [sk],
                    "sd": [sd],
                    "n" : [n],
                    "seed": [seed]
        }
        new_results = pd.DataFrame(new_results_raw)
        new_results = new_results.astype({'sp' : str, 
                                          'd' : int, 
                                          'fr' : float, 
                                          'sk' : float, 
                                          'sd' : float,
                                          'n' : int, 
                                          'seed' : int,})
        
        ################################################
        ##  Write to (existing or new) results file
        ################################################

        if os.path.exists(results_filename):
            results_df = pd.concat([results_df, new_results], ignore_index=True)
            result_row_index = results_df.shape[0] - 1 # because we just added the last row and indexing is off by 1
        else:
            print("==> Creating new results file")
            results_df = new_results
            result_row_index = 0 # because this is a df of 1 row


    assert result_row_index is not None, "Result row index was not defined somehow"
    

    ##############################################
    ##  Write any computed extrapolation data to the dataframe
    ##############################################

    temp_df = pd.DataFrame(all_ex_dicts)
    if 'ptidx' in temp_df.columns: # then we ran extrap_with_rad_comp
        for i in range(temp_df.shape[0]-1):
            results_df = pd.concat([results_df, new_results], ignore_index=True) # copy in standard data
        for i in range(temp_df.shape[0]):
            for col in temp_df.columns:
                results_df.at[result_row_index+i, col] = temp_df.iloc[i][col]
    else:
        for extrap_dict in all_ex_dicts:
            for key in extrap_dict:
                results_df.at[result_row_index, key] = extrap_dict[key]

    ##############################################
    ##  Save new data to file
    ##############################################

    if result_row_index is not None:
        print("==> Updated row", result_row_index, ", which is now:")
        print(results_df.iloc[result_row_index])
    else:
        print("==> Extrapolation results df is now:\n", results_df)
    results_df.to_csv(results_filename)

    print("\n")
    
    

    
