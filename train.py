import numpy as np
import pandas as pd
from task import BottleneckTask
from dra import DynamicResourceAllocator
import multiprocessing as mp
from time import time
import datetime

# initializing hyperparameters
n_restarts  = 10
episodes    = 1000
extraEps    = 0
n_stages    = 2
version     = 2
lmda        = 2
sigmaBase   = 50
computeFreq = 50
noPenalty   = True

# Defining key states and actions for bottleneck task
keyStates  = [5, 6, 16, 17] 
keyStates0 = [5, 16]
correctAct = [7, 9, 18, 20]
keyActions = [7, 8, 9, 10, 18, 19, 20, 21]

if n_stages == 3:
    keyStates.extend([27, 28])
    keyStates0.extend([27])
    correctAct.extend([29, 31])
    keyActions.extend([29, 30, 31, 32])


# Defining the function to be run in parallel across models
def train_model(modelType=None, model=None, \
    episodes=episodes, computeFreq=computeFreq,\
    lmda=lmda, n_stages=n_stages, version=version,\
    noPenalty=noPenalty, sigmaBase=sigmaBase):
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
        startEp = 0

        if modelType.split()[0] == 'freq':
            mType = 'freqBased'
            decay = float(modelType.split()[1])

        model = DynamicResourceAllocator(\
            model        = mType,\
            lmda         = lmda,\
            version      = version,\
            episodes     = computeFreq,\
            decay        = decay,\
            n_stages     = n_stages,\
            sigmaBase    = sigmaBase,\
            nGradUpdates = 20,\
            updateFreq   = int(computeFreq/2),\
            printFreq    = 50, \
            printUpdates = False,\
            noPenaltyForSingleActions=noPenalty)
    
    elif (model != None):
        if (modelType != None):
            raise Exception(f'Using model and '\
                f'ignoring modelType.')

        modelType = model.model
        if model.model == 'freqBased':
            if model.decay == 1:
                model.decay = int(1)
            modelType = f'freq '+str(model.decay)
        startEp = model.maxEp
        model.env.stochastic_choice_sets=False

    elif (model == modelType) & (model == None):
        raise ValueError(f'Both modelType and '\
            f'model cannot be None.')

    # initialising columns for dataframe 
    col_model, col_episode, col_state, col_action, \
    col_chAc, col_chAc_w, col_pmistake, col_sigma, \
    col_obj, col_expR, col_cost, col_nvisits,\
    col_pvisit, col_pvisit_w = [], [], [], [], [],\
         [], [], [], [], [], [], [], [], []

    # Training model and collecting everything to plot
    for episode in range(startEp, episodes+startEp, \
                    computeFreq):
        """
        Later on, check whether multiple entries in some
        columns (e.g. cost, obj, expR) per episode lead
        seaborn to plot narrower confidence intervals.
        """
        # Slow exploration only if new model
        if model==None:
            model.beta = min(episode/50, 5)
        
        # Compute expected reward, cost, and objective
        cost = model.computeCost()
        expR = model.computeExpectedReward(1000)
        obj  = expR - cost

        for state in keyStates:
            for a in [0,1]:
                # append everything to plot in cols
                col_model.append(modelType)
                col_episode.append(episode)
                
                col_state.append(state)

                action = model.env.actions[state][a]
                col_action.append(action)

                idx = model.env.idx(state, action)
                col_nvisits.append(model.n_visits[idx])
                col_sigma.append(model.sigma[idx])

                col_obj.append(obj)
                col_expR.append(expR)
                col_cost.append(cost)

                # other action and its index
                a1 = model.env.actions[state][1-a]
                idx1 = model.env.idx(state, a1)
                
                # compute choice accuracy
                chAc = model.n_visits[min(idx,idx1)]\
                    / sum(model.n_visits[[idx,idx1]])

                chAc_w = model.n_visits_w1[min(idx,idx1)]\
                    / sum(model.n_visits_w1[[idx,idx1]])
                
                col_chAc.append(chAc)
                col_chAc_w.append(chAc_w)
                
                # compute prob. of making mistake
                model.env.state = state
                mistakes = 0
                for _ in range(1000):
                    a, _, _, _ = model.act(state)
                    if a not in correctAct:
                        mistakes += 1

                col_pmistake.append(mistakes/1000)
        
        for state in keyStates0:
            idx = model.env.idx(state, state+2)

            n1 = np.sum( model.n_visits[idx:idx+2] )
            n2 = np.sum( model.n_visits[idx+2:idx+4] )

            col_pvisit.append(n1/(n1+n2+1e-5))
            col_pvisit.append(n1/(n1+n2+1e-5))
            col_pvisit.append(n2/(n1+n2+1e-5))
            col_pvisit.append(n2/(n1+n2+1e-5))

            n1w = np.sum( model.n_visits_w1[idx:idx+2] )
            n2w = np.sum( model.n_visits_w1[idx+2:idx+4] )

            col_pvisit_w.append(n1w/(n1w+n2w+1e-5))
            col_pvisit_w.append(n1w/(n1w+n2w+1e-5))
            col_pvisit_w.append(n2w/(n1w+n2w+1e-5))
            col_pvisit_w.append(n2w/(n1w+n2w+1e-5))

        # Train the model
        model.train()

    # End of training
    model.maxEp = episode
    # adding results to model object
    model.results = pd.DataFrame({
        'Model':            col_model,\
        'Episode':          col_episode,\
        'State':            col_state,\
        'Action':           col_action,\
        'Nvisits':          col_nvisits,\
        'P_visit_w':        col_pvisit_w,\
        'P_visit':          col_pvisit,\
        'Sigma':            col_sigma,\
        'Choice accuracy':  col_chAc,\
        'Choice accuracy wtd.': col_chAc_w,\
        'P_mistake':        col_pmistake,
        'Objective':        col_obj,\
        'Expected reward':  col_expR,\
        'Cost':             col_cost        })

    # Printing
    timeTaken = str(datetime.timedelta\
                (seconds=time()-start) )
    print(f'Finished {modelType} in {timeTaken}.')

    return model



def save_results(results, n_stages=n_stages, when='before',
    noPenalty=noPenalty, version=version,):

    # Creating a directory to save the results in
    import os
    import pickle

    # define the name of the directory to be created
    str_append = f'{n_stages}_v' + str(version) + \
        f'_lmda{lmda}_sigBase{sigmaBase}'

    if noPenalty:
        str_append += f'_noPenalty'

    savePath = f"./figures/{str_append}/"

    try:
        os.mkdir(savePath)
    except OSError:
        print ("Creation of the directory %s failed" % savePath)
    else:
        print ("Successfully created the directory %s " % savePath)


    # Concatenating all results in a mega dataframe
    for model in results:
        if 'df' not in locals():
            df = model.results
        else:
            df = pd.concat( [df, model.results], \
                ignore_index=True, sort=False)

    # Saving results in savePath
    fh = open(f'{savePath}df_{when}','wb')
    pickle.dump(df, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    fh = open(f'{savePath}models_{when}','wb')
    pickle.dump(results, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    return savePath



if __name__ == '__main__':
    # Defining model types to train
    modelTypes = ['dra', 'equalPrecision', 'freq 1',
                'freq 0.97', 'freq 0.95', 'freq 0.9']

    # Running in parallel
    pool = mp.Pool()
    results = pool.map(train_model, np.repeat(modelTypes, n_restarts))

    savePath = save_results(results)


    # Retraining after removing stochastic choice sets
    argsToPass = [(None,model,500) for model in results]
    results2 = pool.starmap(train_model, argsToPass)

    savePath = save_results(results2, when='after')