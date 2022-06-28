import pickle
import itertools
import numpy as np
import pandas as pd
from time import time
from experiment import Experiment
import matplotlib.pyplot as plt
import seaborn as sns

lmdas 		= [0.05, 0.1]
sigmaBases 	= [6, 12]
delta_start = [1]

step_sizes      = [1]
n_adaptrials    = [20]
df_list_upper   = []

start = time()

for step_size, n_trials in itertools.product(step_sizes, n_adaptrials):
    
    print(f'\nRunning params step_size={step_size}, adaptTrials={n_trials}.')

    # initializing lists to collect shit
    df_list_lower = []

    for params in itertools.product(lmdas, sigmaBases,delta_start):
        
        print(f'\tlmda, sb, d_start={params}')

        # initializing more lists to collect shit
        l, s, d, ca, m = [], [], [], [], []
        l_mean, s_mean, d_start = [], [], []        
        
        expt = Experiment(lmda=params[0], sigmaBase=params[1], 
                delta_pmt=params[2], varySubjectParams=True, 
                nSubjectsPerModel=4, stepSize_adaptDelta=step_size,
                nTrials_adaptDelta=n_trials, adaptDelta=True)
        
        # modify such that it can take and pass step-size, n_trials
        expt.run()

        # modify such that first (n_trials=) 20 trials are excluded
        choiceDataDF = expt.gatherChoiceData()
        ca = choiceDataDF.groupby('subject-id').agg({'y':'mean'})['y']

        # add results to respective columns
        for model in expt.models:
            l += [model.lmda]
            s += [model.sigmaBase]
            d += [model.env.delta_pmt]
            m += [model.model]
            l_mean  += [params[0]]
            s_mean  += [params[1]]
            d_start += [params[2]]

        # dataframe of results for this experiment
        df_lower = pd.DataFrame({
            'lmda': l,
            'sigmaBase': s,
            'delta_pmt': d,
            'delta_start': d_start,
            'choice accuracy': ca,
            'model-type': m,
            'lmda-mean': l_mean,
            'sb-mean': s_mean
        })
        
        # list of DFs for one value of step-size, n_adaptrials
        df_list_lower += [df_lower]

        print(f'\tTotal time taken = {time()-start} \n')

    # merging DFs and adding other columns
    df = pd.concat(df_list_lower, sort=False, ignore_index=True)
    df['step-size'] = step_size
    df['nTrials-adapt'] = n_trials

    # list of DFs with all results
    df_list_upper += [df]

# merging DF with all results
df_results = pd.concat(df_list_upper, sort=False, ignore_index=True)

# adding extra columns for plotting
df_results['delta-normalized'] = df_results['delta_pmt']/df_results['sigmaBase']
df_results['lmda-binned'] = pd.cut(df_results['lmda'],[0,0.1,0.2])

# plotting to extract important variables

# delta-normalized
for n_trials in n_adaptrials:
    df = df_results.loc[df_results['nTrials-adapt']==n_trials]
    
    f,ax = plt.subplots()
    sns.histplot(data=df, x='delta-normalized', hue='step-size',
        ax=ax, stat='density')
    ax.set_title(f'nTrials_adaptDelta={n_trials}')
    ax.set_xlim([0,2])
    ax.set_ylim([0,0.6])
# plt.show()

for step_size in step_sizes:
    df = df_results.loc[df_results['step-size']==step_size]
    
    f,ax = plt.subplots()
    sns.histplot(data=df, x='delta-normalized', hue='nTrials-adapt',
        ax=ax, stat='density')
    ax.set_title(f'Step size={step_size}')
    ax.set_xlim([0,2])
    ax.set_ylim([0,0.6])
plt.show()

# plotting normalized delta and choice accuracy
f,ax = plt.subplots()
c1 = (df_results['delta_start']==4)

df = df_results.loc[c1]
sns.kdeplot(data=df, x='delta-normalized',y='choice accuracy', 
        hue='lmda-binned',ax=ax)
ax.set_xlim([0,1.5])
ax.set_title(f'Delta-start = 4')
plt.show()




####################################################################
# Model recovery metric:
# - Generate data for N experiments w and w/o adapting Delta
#   - N = 9: lmda = (0.05, 0.1, 0.15) x sigmaBase = (5,10,15)
# - fit posteriors and compute model recovery metric
# - show plots for recovery metrics w and w/o adaptive Delta
####################################################################

from generateFakeData import generateFakeData

n_restarts = 3
paramSet = {
    'lmdas': 		[0.05, 0.15],
    'sigmaBases': 	[5, 10, 15],
    'delta_pmts': 	np.repeat(np.array([1,4,7]), n_restarts)
}

for adaptDelta in [True,False]:
    df_list, expt_list = generateFakeData(**paramSet, varyParams=True)
    
    fh = open(f'../data/simulatedData/choiceDataDF_varyParams_adaptDelta{adaptDelta}_1_list','wb')
    pickle.dump(df_list, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    fh = open(f'../data/simulatedData/experiments_varyParams_adaptDelta{adaptDelta}_1_list','wb')
    pickle.dump(expt_list, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()


import gc
from datetime import timedelta
from stan_code import fit_posterior_and_save

# define model types
modelTypes = ['dra','freq-s','stakes','equalPrecision']

for adaptDelta in [True,False]:
    # savePath
    basePath = f'../data/simulatedData/'
    savePath = f'{basePath}posteriors_varyParams_adaptDelta_{adaptDelta}'

    # load list of dataframes
    fh = open(f'{basePath}/choiceDataDF_varyParams_adaptDelta{adaptDelta}_list','rb')
    df_list = pickle.load(fh)
    fh.close()

    # loop through all dataframes and compute posterior
    # probability of model given data
    for df_idx, df in enumerate(df_list):
        
        if df is not None:
            
            start = time()

            # printing progress
            print(f'\n\n{df_idx+1}/{len(df_list)}:')

            # compute posterior probability of obtaining signature
            fit_posterior_and_save(df, savePath=savePath, expt_id=df_idx, n_samples=2500, n_chains=4)
            gc.collect()

            print(f'\nTime taken: {timedelta(seconds=(time()-start))}')