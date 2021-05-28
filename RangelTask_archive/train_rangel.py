import numpy as np
import pandas as pd
from task import RangelTask
from dra_rangel import DynamicResourceAllocator
from scipy.stats import norm
import multiprocessing as mp
from time import time
import datetime

# initializing hyperparameters
n_restarts  = 5
lmda        = 0.1
sigmaBase   = 5
delta_1     = 4
delta_2     = 1
delta_pmt   = 2.5
learnPMT    = False

# i could define and pass **hyperparams as a dict
hyperparams = {
    'lmda'      : 0.1,
    'sigmaBase' : 5,
    'delta_1'   : 4,
    'delta_2'   : 1,
    'delta_pmt' : 2.5,
    'learnPMT'  : False }


# Defining the function to be run in parallel across models
def train_model(modelType='dra', lmda=lmda,\
    sigmaBase=sigmaBase, learnPMT=learnPMT,\
    delta_pmt=delta_pmt, delta_1=delta_1, delta_2=delta_2):
    """
    Trains model = {modelType} for {episodes} episodes
    with stochastic choice sets, then for an additional
    {extraEps} episodes without stochastic choice sets in
    the Bottleneck task. Computes numerous things to plot
    every {computeFreq} episodes and stores them in a
    dataframe.

    Returns the trained model that has the dataframe of
    results stored in model.results.
    """
    
    # Starting time counter
    start = time()

    # Define the model    
    model = DynamicResourceAllocator(\
        model        = modelType,\
        lmda         = lmda,\
        sigmaBase    = sigmaBase,\
        delta_1      = delta_1,\
        delta_2      = delta_2,\
        delta_pmt    = delta_pmt,\
        printUpdates = False,\
        learnPMT     = learnPMT)
    
    # Train the model
    model.train()

    # Printing
    timeTaken = str(datetime.timedelta\
                (seconds=time()-start) )
    print(f'Finished {modelType} in {timeTaken}.')

    return model



def save_results(results, lmda=lmda, delta_1=delta_1, delta_2=delta_2, delta_pmt=delta_pmt, learnPMT=learnPMT, sigmaBase=sigmaBase, append_str=''):

    # Creating a directory to save the results in
    import os
    import pickle

    # define the name of the directory to be created
    folderName = f'delta{delta_1}{delta_2}_learnPMT{int(learnPMT)}_Delta{delta_pmt}_lmda{lmda}_sigBase{sigmaBase}_episodes{results[0].env.episodes}{append_str}'

    savePath = f"./figures/rangelTask/{folderName}/"

    try:
        os.mkdir(savePath)
    except OSError:
        print ("Creation of the directory %s failed" % savePath)
    else:
        print ("Successfully created the directory %s " % savePath)


    # Concatenating all results in a mega dataframe
    for model in results:
        if 'df' not in locals():
            df = model.memoryTable
            df = df[df['action']<12]
            df['model'] = model.model
        else:
            df_new = model.memoryTable
            df_new = df_new[df_new['action']<12]
            df_new['model'] = model.model
            df = pd.concat( [df, df_new], \
                ignore_index=True, sort=False)

    # Saving results in savePath
    fh = open(f'{savePath}df','wb')
    pickle.dump(df, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    fh = open(f'{savePath}models','wb')
    pickle.dump(results, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    return savePath



if __name__ == '__main__':
    # Defining model types to train
    n_restarts = 10
    sigmaBase  = 5
    modelTypes = ['dra', 'freq-sa', 'freq-s', 'stakes', 'equalPrecision']

    # Running in parallel
    pool = mp.Pool()
    lmda = 0.1
    delta_2 = 1
    delta_1 = 4
    delta_pmt = 2
    learnPMT = False
    sigmaBase = 5
    argsToPass = [(modelType, lmda, sigmaBase, learnPMT, delta_pmt, delta_1, delta_2) for modelType in modelTypes*n_restarts]

    results = pool.starmap(train_model, argsToPass)
    # results = pool.map(train_model, np.repeat(modelTypes, n_restarts))
    
    # # Run models in series
    # for lmda in [0.1, 0.05, 0.2]:
    #     for (delta_1,delta_2) in [(4,2),(4,1)]:
    #         for delta_pmt in [1.5, 2.5]:
    #             for learnPMT in [False,True]:
                    
    #                 results = []

    #                 for modelType in modelTypes:
                    
    #                     for _ in range(n_restarts):
    #                         model = train_model(modelType=modelType, learnPMT=learnPMT, delta_pmt=delta_pmt, delta_1=delta_1, delta_2=delta_2, lmda=lmda)
    #                         results.append(model)
                    
    #                 savePath = save_results(results, lmda=lmda, delta_1=delta_1, delta_2=delta_2, delta_pmt=delta_pmt, learnPMT=learnPMT, sigmaBase=sigmaBase)
    #                 print(f'\n\n Finished training all models for lmda={lmda}, delta=({delta_1},{delta_2}), Delta={delta_pmt}, learnPMT={learnPMT}.\n\n')

    # # Retraining after removing stochastic choice sets
    # savePath = save_results(results2)