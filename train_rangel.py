import numpy as np
import pandas as pd
from task import RangelTask
from dra_rangel import DynamicResourceAllocator
import multiprocessing as mp
from time import time
import datetime

# initializing hyperparameters
n_restarts  = 5
lmda        = 0.1
sigmaBase   = 5
delta_1     = 4
delta_2     = 2
delta_pmt   = 1.5
learnPMT    = False

# i could define and pass **hyperparams as a dict
hyperparams = {
    'lmda'      : 0.1,
    'sigmaBase' : 5,
    'delta_1'   : 4,
    'delta_2'   : 2,
    'delta_pmt' : 1.5,
    'learnPMT'  : False }


# Defining the function to be run in parallel across models
def train_model(modelType=None, model=None, lmda=lmda,\
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
    
    if (model==None) & (modelType is not None):
        # Splitting freqBased models into type and decay
        mType = modelType
        decay = 1

        if modelType.split()[0] == 'freq':
            mType = 'freqBased'
            decay = float(modelType.split()[1])

        model = DynamicResourceAllocator(\
            model        = mType,\
            lmda         = lmda,\
            sigmaBase    = sigmaBase,\
            delta_1      = delta_1,\
            delta_2      = delta_2,\
            delta_pmt    = delta_pmt,\
            printUpdates = False,\
            learnPMT     = learnPMT)
    
    elif (model != None):
        if (modelType != None):
            raise Exception(f'Using model and '\
                f'ignoring modelType.')

        modelType = model.model
        if model.model == 'freqBased':
            if model.decay == 1:
                model.decay = int(1)
            modelType = f'freq '+str(model.decay)
        # model.env.stochastic_choice_sets=False

    elif (model == modelType) & (model == None):
        raise ValueError(f'Both modelType and '\
            f'model cannot be None.')

    # Train the model
    # return choices on PMT trials here!
    model.train()

    # Printing
    timeTaken = str(datetime.timedelta\
                (seconds=time()-start) )
    print(f'Finished {modelType} in {timeTaken}.')

    return model



def save_results(results):

    # Creating a directory to save the results in
    import os
    import pickle

    # define the name of the directory to be created
    str_append = f'learnPMT{int(learnPMT)}_delta{delta_1}{delta_2}_Delta{delta_pmt}_lmda{lmda}_sigBase{sigmaBase}'

    savePath = f"./figures/rangelTask/{str_append}/"

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
    n_restarts = 5
    modelTypes = ['dra', 'equalPrecision', 'freq 1']

    # # Running in parallel
    # pool = mp.Pool()
    # results = pool.map(train_model, np.repeat(modelTypes, n_restarts))
    
    # Run models in series
    for lmda in [0.1, 0.05, 0.2]:
        for (delta_1,delta_2) in [(4,2),(4,1)]:
            for delta_pmt in [1.5, 2.5]:
                for learnPMT in [False,True]:
                    
                    results = []

                    for modelType in modelTypes:
                    
                        for _ in range(n_restarts):
                            model = train_model(modelType=modelType, learnPMT=learnPMT, delta_pmt=delta_pmt, delta_1=delta_1, delta_2=delta_2, lmda=lmda)
                            results.append(model)
                    
                    savePath = save_results(results)
                    print(f'\n\n Finished training all models for lmda={lmda}, delta=({delta_1},{delta_2}), Delta={delta_pmt}, learnPMT={learnPMT}.\n\n')

    # # Retraining after removing stochastic choice sets
    # argsToPass = [(None,model,500) for model in results]
    # results2 = pool.starmap(train_model, argsToPass)

    # savePath = save_results(results2)