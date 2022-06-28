import gc
import pickle
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from time import time
from datetime import timedelta
from stan_code import fit_posterior_and_save

# load list of dataframes
fh = open(f'./simulatedData/choiceDataDFvaryParams_list','rb')
df_list = pickle.load(fh)
fh.close()

# # initialize variables to save
# posteriors_list  = []
# df_postProb_list = []
# modelRecoveryMatrix_list = []

# last experiment processed before crashing
expt_id = 591

# define model types
modelTypes = ['dra','freq-s','stakes','equalPrecision']
th_factors = list(range(4))

# loop through all dataframes and compute posterior
# probability of model given data
for df_idx, df in enumerate(df_list[expt_id:600]):
    
    if df is not None:
        
        start = time()
        df_idx += expt_id

        # printing progress
        print(f'\n\n{df_idx+1}/{len(df_list)}:')

        # # record lmda, sigmaBase, Delta_PMT
        # lmda      = float(df['lmda'].unique())
        # sigmaBase = float(df['sigmaBase'].unique())
        # delta_pmt = float(df['delta_pmt'].unique())

        # compute posterior probability of obtaining signature
        fit_posterior_and_save(df, df_idx, n_samples=2500, n_chains=4)
        gc.collect()

        print(f'\nTime taken: {timedelta(seconds=(time()-start))}')

    # add to the list of posteriors
    # posteriors_list += [posterior]

    # # initialize lists
    # modelRecovery_fro = []
    # modelRecovery_det = []

    # # vary threshold for tests
    # for th_factor in th_factors:

    #     # run tests to check probability of predefinted model signatures
    #     # using default threshold 3*SD
    #     modelRecoveryMatrix = np.identity(len(modelTypes))
    #     for i in range(len(modelTypes)):
    #         modelTest = np.around(prob_of_signature(posterior[i]['mu_ca'], th_factor), 2)
    #         modelRecoveryMatrix[i] = modelTest

    #     # calculate Frobenius distance from id matrix
    #     modelRecovery_fro += [np.linalg.norm(modelRecoveryMatrix-np.identity(len(modelTypes)), ord='fro')]
    #     modelRecovery_det += [np.linalg.det(modelRecoveryMatrix)]
    #     modelRecoveryMatrix_list += [modelRecoveryMatrix]

    # # store all test factors in dataframe
    # df_postProb_expt = pd.DataFrame({
    #     'lmda':         [lmda] * len(th_factors),
    #     'sigmaBase':    [sigmaBase] * len(th_factors),
    #     'delta_pmt':    [delta_pmt] * len(th_factors),
    #     'thresholdFac': th_factors,
    #     'modelRec-Fro': modelRecovery_fro,
    #     'modelRec-Det': modelRecovery_det
    # })

    # # save dataframe in list of all dataframes 
    # # (to be combined later: may throw memory error)
    # df_postProb_list += [df_postProb_expt]

    # print(f'\nTime taken: {timedelta(seconds=(time()-start))}')


# # save posteriors and dataframes with model recovery
# fh = open(f'./simulatedData/posteriors_list_{int(expt_id)}-{int(expt_id+44)}','wb')
# pickle.dump(posteriors_list, fh, pickle.HIGHEST_PROTOCOL)
# fh.close()

# del posteriors_list
# posteriors_list = []

# fh = open(f'./simulatedData/df_modelPosteriorProbability_list_{int(expt_id)}-{int(expt_id+44)}','wb')
# pickle.dump(df_postProb_list, fh, pickle.HIGHEST_PROTOCOL)
# fh.close()

# fh = open(f'./simulatedData/modelRecoveryMatrix_list_{int(expt_id)}-{int(expt_id+44)}','wb')
# pickle.dump(modelRecoveryMatrix_list, fh, pickle.HIGHEST_PROTOCOL)
# fh.close()
