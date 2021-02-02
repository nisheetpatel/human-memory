import numpy as np
import pandas as pd
from task import BottleneckTask
from dra import DynamicResourceAllocator
import multiprocessing as mp
from time import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# initializing hyperparameters
n_restarts  = 5
episodes    = 1000
extraEps    = 501
n_stages    = 3

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
def train_model(modelType, lmda=1, episodes=episodes,\
    extraEps=extraEps, computeFreq=20, n_stages=3):
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

    # Splitting freqBased models into type and decay
    mType = modelType
    decay = 1
    if modelType.split()[0] == 'freq':
        mType = 'freqBased'
        decay = float(modelType.split()[1])

    model = DynamicResourceAllocator(model=mType, \
        lmda=1, episodes=computeFreq, decay=decay, \
        nGradUpdates=20, updateFreq=int(computeFreq/2),\
        printFreq=1, printUpdates=False, n_stages=n_stages)

    # initialising columns for dataframe 
    col_model, col_episode, col_state, col_action, \
    col_choiceAccuracy, col_pmistake, col_sigma, \
    col_obj, col_expR, col_cost, \
    col_nvisits, col_pvisit = [], [], [], [], [],\
         [], [], [], [], [], [], []

    old_nvisits = np.zeros(len(model.n_visits))

    # Training model and collecting everything to plot
    for episode in range(0, episodes+extraEps, \
                    computeFreq):
        """
        Later on, check whether multiple entries in some
        columns (e.g. cost, obj, expR) per episode lead
        seaborn to plot narrower confidence intervals.
        """
        
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
                
                col_choiceAccuracy.append(chAc)
                
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

            n1 = np.sum( model.n_visits[idx:idx+2] ) -\
                 np.sum( old_nvisits[idx:idx+2] )

            n2 = np.sum( model.n_visits[idx+2:idx+4] ) -\
                 np.sum( old_nvisits[idx+2:idx+4] )

            col_pvisit.append(n1/(n1+n2+1e-5))
            col_pvisit.append(n1/(n1+n2+1e-5))
            col_pvisit.append(n2/(n1+n2+1e-5))
            col_pvisit.append(n2/(n1+n2+1e-5))

        # Store old n_visits to compute prop. visits
        old_nvisits = model.n_visits.copy()

        # See whether the environment needs to be switched
        timeToSwitch = (episode >= episodes) & \
            (episode < episodes+computeFreq)

        if timeToSwitch:
            # Changing env to remove stoch. choice sets
            model.env = BottleneckTask(n_stages=n_stages,\
                stochastic_choice_sets=False)
        
        # Train the model
        model.train()

    # End of training
    # adding results to model object
    model.results = pd.DataFrame({
        'Model':            col_model,\
        'Episode':          col_episode,\
        'State':            col_state,\
        'Action':           col_action,\
        'Nvisits':          col_nvisits,\
        'P_visit':          col_pvisit,\
        'Sigma':            col_sigma,\
        'Choice accuracy':  col_choiceAccuracy,\
        'P_mistake':        col_pmistake,
        'Objective':        col_obj,\
        'Expected reward':  col_expR,\
        'Cost':             col_cost        })

    # Printing
    timeTaken = str(datetime.timedelta\
                (seconds=time()-start) )
    print(f'Finished {modelType} in {timeTaken}.')

    return model



# Defining model types to train
modelTypes = ['dra', 'equalPrecision', 'freq 1',\
             'freq 0.9', 'freq 0.75', 'freq 0.5']

# Running in parallel
pool = mp.Pool()
results = pool.map(train_model, \
    np.repeat(modelTypes, n_restarts))


# Concatenating all results in a mega dataframe
for model in results:
    if 'df' not in locals():
        df = model.results
    else:
        df = pd.concat( [df, model.results], \
            ignore_index=True, sort=False)


# Creating a directory to save the results in
import os

# define the name of the directory to be created
savePath = f"./figures/{n_stages}"

try:
    os.mkdir(savePath)
except OSError:
    print ("Creation of the directory %s failed" % savePath)
else:
    print ("Successfully created the directory %s " % savePath)


# Plotting Objective, expected reward, cost
for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df, x="Episode", y=toPlot,\
        hue="Model", style="Model", dashes=False)
    plt.savefig(f'{savePath}Perf_{toPlot}')
    plt.show()


# Plotting evolution of observables
for modelType in modelTypes:
    
    # Time updates
    start = time()

    # Extracting model's data
    dfModel = df[df['Model']==modelType]

    if modelType.split()[0] == 'freq':
        saveName = 'freq' + str(int((float\
            (modelType.split()[1])*100)))
    else:
        saveName = modelType.split()[0]

    # Plotting p visit
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=dfModel, x="Episode", \
        y="P_visit", hue="State",\
        legend='full', palette="Set2")
    ax.set_title(f'{modelType}')
    ax.set_ylim([0,1])
    plt.savefig(f'{savePath}pvisit_{saveName}')
    plt.show()

    # Plotting sigma
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=dfModel, x="Episode", \
        y='Sigma', hue='Action', ci=None,\
        legend='full', palette="Paired")
    ax.set_title(f'{modelType}')
    ax.set_ylim([0,55])
    plt.savefig(f'{savePath}sigma_{saveName}')
    plt.show()

    # Plotting MA of P(mistake)
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=dfModel, x="Episode", \
        y="P_mistake", hue="State",\
        legend='full', palette="Paired")
    ax.set_title(f'{modelType}')
    ax.set_ylim([0,0.65])
    plt.savefig(f'{savePath}Obs_Pmistake_{saveName}')
    plt.show()

    # Plotting choice accuracy
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=dfModel, x="Episode",\
        y="Choice accuracy", hue="State",\
        legend='full', palette="Paired")
    ax.set_title(f'{modelType}')
    ax.set_ylim([0.5,1])
    plt.savefig(f'{savePath}Obs_choiceAccuracy_{saveName}')
    plt.show()

    # Printing time updates
    print(f'Time to plot for {modelType} = '\
     f'{str(datetime.timedelta(seconds=time()-start))}')

plt.close('all')