import pickle
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from train_rangel import train_model, save_results

for expt in range(50):
    # Training all models and saving results
    # Defining model types to train
    n_restarts = 10
    modelTypes = ['dra', 'freq-sa', 'freq-s', 'stakes', 'equalPrecision']

    # Defining parameters   
    #   - Currently has 288 parameter sets
    #   - should take 2-2.5 hours to run 3 models; 3-3.5 hours to run 5 models
    lmda = [0.1]            # [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    delta_1 = [4]
    delta_2 = [1]           # [0.5, 1, 2]
    delta_pmt = [2]         # [1,2,3,4]
    learnPMT = [False]
    sigmaBase = [5]         # [3,5,7,9]
    params = itertools.product(lmda, sigmaBase, learnPMT, delta_pmt, delta_1, delta_2)

    # Running in parallel
    pool = mp.Pool()
    for argsToPass in params:
        modelArgs = []
        for modelType in np.repeat(modelTypes,n_restarts):
            modelArg = [modelType]
            modelArg.extend(list(argsToPass))
            modelArgs.append(tuple(modelArg))
        results = pool.starmap(train_model, modelArgs)
        lmda, sigmaBase, learnPMT, delta_pmt, delta_1, delta_2 = argsToPass
        savePath = save_results(results, lmda, delta_1, delta_2, delta_pmt, learnPMT, sigmaBase, append_str=f'_expt{expt}')



# Importing plotting libraries
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set2')
#sns.set()

# Plotting
# Cycle through directories
pathlist = Path(f'./figures/rangelTask/').glob('*')
modelTypes = ['dra', 'freq-sa', 'freq-s', 'stakes', 'equalPrecision']
# modelTypes = ['dra', 'equalPrecision', 'freqBased']

for path in pathlist:
    # load models and dfs
    models = pickle.load(open(f'./{str(path)}/models','rb'))
    df = pickle.load(open(f'./{str(path)}/df','rb'))

    # # average across actions
    # df['run'] = np.tile( np.repeat(np.arange(12), n_restarts), len(modelTypes))
    # df = df.groupby(['model','state','run']).agg({'sigma': 'mean'}).reset_index()
    # df.model = df.model.astype("category")
    # df.model.cat.set_categories(modelTypes, inplace=True)
    # df.sort_values(['model'], inplace=True)

    # plot sigmas of model in each state
    sns.lineplot(data=df, x='state', y='sigma', hue='model')
    plt.xticks([0, 1, 2, 3], ['s1', 's2', 's3', 's4'])
    plt.ylim([1.5, 5.2])
    plt.savefig(f'{str(path)}/sigma.svg')
    plt.savefig(f'{str(path)}/sigma.png')
    plt.close()


    # Compute probability that the model picks deterministic option
    new_models = []
    dataTypes = ['data','model']

    for model in models:
        model.p_plusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])
        model.p_minusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])

        model.p_plusDelta_true = np.zeros(12)
        model.p_minusDelta_true = np.zeros(12)
        for a in range(12):
            # +Delta
            idx = (np.where((model.env.pmt_trial>0) & (model.choicePMT==a)))[0] 
            model.p_plusDelta_true[a] = 1-len(idx)/(model.env.n_pmt/2)
            
            # -Delta
            idx = (np.where((model.env.pmt_trial<0) & (model.choicePMT==a)))[0]  
            model.p_minusDelta_true[a] = len(idx)/(model.env.n_pmt/2)

        # Create empty dataframe with necessary columns
        model.df_p = pd.DataFrame({
            'state':                    [],
            'action':                   [],
            'p(choose better option)':  [],
            'type':                     [],
            'pmt-type':                 []
        })

        for dataType in dataTypes:
            for pmtType in ['+','-']:
                if (dataType=='data'):
                    if pmtType=='+':
                        pCol = model.p_plusDelta_true
                    else:
                        pCol = model.p_minusDelta_true
                else:
                    if pmtType=='+':
                        pCol = model.p_plusDelta
                    else:
                        pCol = model.p_minusDelta

                typeCol = [dataType] * 12
                pmtCol  = [pmtType] * 12
                
                df_p_specific = pd.DataFrame( {
                    'state':                np.repeat(['s1','s2','s3','s4'],3),\
                    'action':               model.env.actions[:12],
                    'p(choose better option)':pCol,
                    'type':                 typeCol,
                    'pmt-type':             pmtCol  })
                
                model.df_p = pd.concat([model.df_p, df_p_specific], ignore_index=True, sort=False)
        
        model.df_p['model'] = model.model

        new_models.append(model)

    # Concatenate results for all models into one df
    for model in new_models:
        if 'df_p' not in locals():
            df_p = model.df_p
        else:
            df_p_new = model.df_p
            df_p = pd.concat( [df_p, df_p_new], \
                ignore_index=True, sort=False)


    # Plotting p(choose better option) on PMT trials 
    # based on data collected during those and model's learnt sigma
    for dataType in dataTypes:
        f,ax=plt.subplots()
        sns.lineplot(data=df_p[df_p['type']==dataType], x='state', y='p(choose better option)', hue='model')
        plt.ylim([.5,1])
        plt.title(f'{dataType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{dataType}.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{dataType}.png')
        plt.close()


    # Plotting p(choose better option) on PMT trials in the space of differences
    stateDiff, actions, pChooseBetterDiff, dTypeCol, modelCol = [], [], [], [], [] 
    for modelType in modelTypes: 
        for dataType in dataTypes:
            for (si,sj) in [('s1','s2'),('s3','s4'),('s1','s3'),('s2','s4'),('s1','s4')]: 
                for a in [0,1,2]: 
                    
                    p_si_a = df_p.loc [ (df_p['state']==si) & (df_p['action']%3==a) & (df_p['model']==modelType) & (df_p['type']==dataType), 'p(choose better option)']
                    p_sj_a = df_p.loc [ (df_p['state']==sj) & (df_p['action']%3==a) & (df_p['model']==modelType) & (df_p['type']==dataType), 'p(choose better option)']
                    pDiff = np.array(p_si_a) - np.array(p_sj_a)
                    
                    pChooseBetterDiff.extend( list(pDiff) ) 
                    
                    actions.extend( [a] * len(pDiff) ) 
                    stateDiff.extend( [f'{si}-{sj}'] * len(pDiff) ) 
                    
                    dTypeCol.extend( [dataType] * len(pDiff) ) 
                    modelCol.extend( [modelType] * len(pDiff) ) 
                    
    df_p_diff = pd.DataFrame( { 
                    'state':                stateDiff, 
                    'action':               actions, 
                    'p(choose better option)':pChooseBetterDiff, 
                    'type':                 dTypeCol, 
                    'model':                modelCol})

    for dataType in dataTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p_diff[df_p_diff['type']==dataType], x='state', y='p(choose better option)', hue='model')
        plt.title(dataType)
        plt.ylim([-0.5, 0.5])
        plt.savefig(f'{str(path)}/p_diff_{dataType}.svg')
        plt.savefig(f'{str(path)}/p_diff_{dataType}.png')
        plt.close()

    # Saving these dataframes
    f1 = open(f'./{path}/df_p','wb')
    pickle.dump(df_p, f1, pickle.HIGHEST_PROTOCOL)
    f1.close()
    f2 = open(f'./{path}/df_p_diff','wb')
    pickle.dump(df_p_diff, f2, pickle.HIGHEST_PROTOCOL)
    f2.close()
    
    del df_p, df_p_new, df, df_p_diff


# Plotting variance of mean across experiments
# in normal space
if 'df' in locals():
    del df
pathlist = Path(f'./figures/rangelTask/Params_0.150241_1530_50expts/').glob('*')
modelTypes = ['dra', 'freq-sa', 'freq-s', 'stakes', 'equalPrecision']

for path in pathlist:
    df_p = pickle.load(open(f'./{str(path)}/df_p','rb'))
    df_p['expt'] = str(path)[-5:]

    if 'df' not in locals():
        df = df_p.groupby(['expt','model','type','state'])\
                      .agg({'p(choose better option)': 'mean'})\
                      .reset_index()
    else:
        df_new = df_p.groupby(['expt','model','type','state'])\
                          .agg({'p(choose better option)': 'mean'})\
                          .reset_index()
        df = pd.concat( [df, df_new], \
            ignore_index=True, sort=False)

# sorting df according to column model (for consistent color scheme)
df.model = df.model.astype("category") 
df.model.cat.set_categories(modelTypes, inplace=True)
df.sort_values(['model'])

for dataType in dataTypes: 
    f,ax=plt.subplots() 
    sns.boxplot(data=df[df['type']==dataType], x='state', y='p(choose better option)', hue='model') 
    plt.title(dataType) 
    plt.title(dataType)


# Plotting variance of mean across experiments
# in space of differences
if 'df' in locals():
    del df
pathlist = Path(f'./figures/rangelTask/').glob('*')
modelTypes = ['dra', 'freq-sa', 'freq-s', 'stakes', 'equalPrecision']

for path in pathlist:
    df_p_diff = pickle.load(open(f'./{str(path)}/df_p_diff','rb'))
    df_p_diff['expt'] = str(path)[-5:]

    if 'df' not in locals():
        df = df_p_diff.groupby(['expt','model','type','state'])\
                      .agg({'p(choose better option)': 'mean'})\
                      .reset_index()
    else:
        df_new = df_p_diff.groupby(['expt','model','type','state'])\
                          .agg({'p(choose better option)': 'mean'})\
                          .reset_index()
        df = pd.concat( [df, df_new], \
            ignore_index=True, sort=False)

# sorting df according to column model (for consistent color scheme)
df.model = df.model.astype("category") 
df.model.cat.set_categories(modelTypes, inplace=True)
df.sort_values(['model'])

for dataType in dataTypes: 
    f,ax=plt.subplots() 
    sns.boxplot(data=df[df['type']==dataType], x='state', y='p(choose better option)', hue='model') 
    plt.title(dataType) 
    plt.title(dataType)