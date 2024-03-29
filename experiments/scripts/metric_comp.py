# Data quality repo
#   Metric computation routines
#   Saves to file zz_results_df.csv

import numpy as np
from optparse import OptionParser
import os
import pandas as pd
import interpml
from data_metrics import *
from data_gen import *


if __name__ == "__main__":
    """ Driver code """

    usage = "%prog [options]"
    parser = OptionParser(usage)
    parser.add_option("--infile", dest="infile", type=str, default=None,
        help="Path to .npy file to read in.")
    
    (options, args) = parser.parse_args()

    assert os.getcwd()[-7:] == 'scripts', "Run metric_comp.py from the scripts sub-directory."

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

    # set to False to check effect of not rescaling
    rescalex = True
    rescaley = True

    ################################################
    ##  Check if already ran metrics on this dataset
    ################################################

    result_row_index = None
    results_filename = os.path.dirname(options.infile) + '/zz_results_df.csv' # results file in same directory as data
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
            print("Found matching row - exiting") # <=== NOT !!!")
            exit()
        elif len(matching_rows) > 1:
            print("==> WARNING: duplicate rows exist for same data sample; exiting")
            exit()
        else:
            pass # a row for this datasset has not been created in the results file

    
    ##############################################
    ##  Sampling density metric computation
    ##############################################

    print("====> Computing sampling density.")
    dens = samplingDensity(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'dens'] = dens
    

    ##############################################
    ##  Discrepancy metric computation; transform to [0,1]^d first
    ##############################################

    # disc = 999 # only keep this line if discrepancy computation is commented out
    
    print("====> Computing discrepancy.")
    X_to_01 = 0.5*(X + np.ones(X.shape[1]))
    disc = discrepancy(X_to_01)
    if result_row_index is not None:
        results_df.at[result_row_index, 'disc'] = disc   


    ##############################################
    ##  maximin metric computation
    ##############################################

    # mxmin = 999 # only keep this line if maximin computation is commented out
    
    print("====> Computing maximin.")
    mxmin = maximin(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'mxmin'] = mxmin  

    ##############################################
    ##  minimax metric computation
    ##############################################

    # mnmax = 999 # only keep this line if minimax computation is commented out
    
    print("====> Computing minimax.")
    mnmax = minimax(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'mnmax'] = mnmax 

    ##############################################
    ##  sampling distances metric computation
    ##############################################

    # smpDist = 999 # only keep this line if sampling distances computation is commented out
    
    print("====> Computing sampling distances.")
    smpDist = samplingDistances(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'smpDist'] = smpDist


    ##############################################
    ##  E-optimality metric computation
    ##############################################

    # eOpt = 999 # only keep this line if sampling distances computation is commented out
    
    print("====> Computing E-optimality (min eigenvalue of linear Fisher info matrix).")
    eOpt = eOptimality(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'eOpt'] = eOpt


    ##############################################
    ##  covariance condition number metric computation
    ##############################################

    # covCon = 999 # only keep this line if covariance condition computation is commented out
    
    print("====> Computing condition number of the covariance matrix.")
    covCon = covarianceCondition(X)
    if result_row_index is not None:
        results_df.at[result_row_index, 'covCon'] = covCon


    ##############################################
    ##  Evaluate error bounds at a point
    ##############################################
 
    # use a random point at radius eval_rad defined below
    # note: expect higher data density at radius sqrt(dim)

    def eval_at_radius(eval_rad):
        dir_vec = rng.normal(size=(X.shape[1]))
        norm_vec = dir_vec/np.linalg.norm(dir_vec)
        eval_pt = eval_rad * norm_vec

        err_dict = {} # empty dictionary to hold computed errors

        print("====> Computing Delaunay estimate at point with radius",eval_rad)
        err_dict['delaunayErrEst'+str(eval_rad)] = interpml.delaunayError(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley)
        print("====> Computing RBF estimate at point with radius",eval_rad)
        err_dict['rbfErrEst'+str(eval_rad)] = interpml.rbfError(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley)
        print("====> Computing GP estimate at point with radius",eval_rad)
        err_dict['gpErrEst'+str(eval_rad)] = interpml.gpError(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley) 

        # # create a sampler class to access the response function rf
        my_sampler = Sampler(rng, d, frequency=fr, skewness=sk, std_dev=sd)

        eval_y = my_sampler.rf(eval_pt)
        eval_y = np.array(eval_y).reshape(1)
        print("====> Computed  actual value", eval_y, "at above point")
        print("====> Computing Delaunay error")
        err_dict['delaunayIntError'+str(eval_rad)] = np.abs(interpml.delaunayInterp(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley) - eval_y)
        print("====> Computing rbf error")
        err_dict['rbfIntError'+str(eval_rad)] = np.abs(interpml.rbfInterp(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley) - eval_y)
        print("====> Computing gp error")
        err_dict['gpIntError'+str(eval_rad)] = np.abs(interpml.gpInterp(X, Y, eval_pt,rescale_x=rescalex, rescale_y=rescaley) - eval_y)
        print("====> Computing ReLU MLP regressor error")
        err_dict['mlpIntErr'+str(eval_rad)] = np.abs(interpml.reluMlpInterp(X, Y, eval_pt, use_valid=False,rescale_x=rescalex, rescale_y=rescaley) - eval_y)
        # print("====> Computing random forest regressor error")
        # err_dict['rfIntErr'+str(eval_rad)] = np.abs(interpml.rfInterp(X, Y, eval_pt) - eval_y)
   
        return err_dict


    all_rads = [0.1, 1.0, 2.0]
    all_dicts = []

    for rad in all_rads:
        err_dict = eval_at_radius(rad) # returns a dictionary
        all_dicts.append(err_dict)


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
                    "seed": [seed],
                    "dens": [dens],
                    "disc": [disc],
                    'mxmin': [mxmin],
                    'mnmax': [mnmax],
                    'eOpt': [eOpt],
                    'covCon': [covCon],
        }
        new_results = pd.DataFrame(new_results_raw)
        new_results = new_results.astype({'sp' : str, 
                                          'd' : int, 
                                          'fr' : float, 
                                          'sk' : float, 
                                          'sd' : float,
                                          'n' : int, 
                                          'seed' : int,
                                          'dens' : float,
                                          'disc' : float,
                                          'mxmin': float,
                                          'mnmax': float,
                                          'eOpt': float,
                                          'covCon': float})
        
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
    ##  Write any computed errors to the dataframe
    ##############################################

    for err_dict in all_dicts:
        for key in err_dict:
            results_df.at[result_row_index, key] = err_dict[key]

    ##############################################
    ##  Save new data to file
    ##############################################

    if result_row_index is not None:
        print("==> Updated row", result_row_index, ", which is now:")
        print(results_df.iloc[result_row_index])
    else:
        print("==> Results df is now:\n", results_df)
    results_df.to_csv(results_filename)

    print("\n")

    
