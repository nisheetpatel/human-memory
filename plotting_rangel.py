import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from train_rangel import train_model, save_results


# Training all models and saving results
# Defining model types to train
n_restarts = 10
modelTypes = ['dra', 'equalPrecision', 'freq 1']

# # Running in parallel
# pool = mp.Pool()
# results = pool.map(train_model, np.repeat(modelTypes, n_restarts))

# Run models in series
for lmda in [0.1]: #, 0.05, 0.2]:
    for (delta_1,delta_2) in [(4,1)]:
        for delta_pmt in [2]:
            for learnPMT in [False,True]:
                sigmaBase = 5
                results = []

                for modelType in modelTypes:
                
                    for _ in range(n_restarts):
                        model = train_model(modelType=modelType, learnPMT=learnPMT, delta_pmt=delta_pmt, delta_1=delta_1, delta_2=delta_2, lmda=lmda, sigmaBase=sigmaBase)
                        results.append(model)
                
                savePath = save_results(results, lmda=lmda, delta_1=delta_1, delta_2=delta_2, delta_pmt=delta_pmt, learnPMT=learnPMT, sigmaBase=sigmaBase)
                print(f'\n\n Finished training all models for lmda={lmda}, delta=({delta_1},{delta_2}), Delta={delta_pmt}, learnPMT={learnPMT}.\n\n')


# Importing more libraries
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

# Plotting
# Cycle through directories
pathlist = Path(f'./figures/rangelTask/').glob('*')
modelTypes = ['dra', 'equalPrecision', 'freqBased']

for path in pathlist:
    # load models and dfs
    models = pickle.load(open(f'{str(path)}/models','rb'))
    df = pickle.load(open(f'{str(path)}/df','rb'))

    # plot and save sigmas
    sns.lineplot(data=df, x='state', y='sigma', hue='model')
    plt.xticks([0, 1, 2, 3], ['s1', 's2', 's3', 's4'])
    plt.ylim([1.5, 5.2])
    plt.savefig(f'{str(path)}/sigma.svg')
    plt.savefig(f'{str(path)}/sigma.png')
    plt.close()

    # Compute probability that the model picks deterministic option
    new_models = []
    for model in models:
        model.p_plusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])
        model.p_minusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])

        model.p_plusDelta_true = np.zeros(12)
        model.p_minusDelta_true = np.zeros(12)
        for a in range(12):
            # +Delta
            idx = (np.where((model.env.pmt_trial>0) & (model.choicePMT==a)))[0] 
            model.p_plusDelta_true[a] = len(idx)/(model.env.n_pmt/2)
            
            # -Delta
            idx = (np.where((model.env.pmt_trial<0) & (model.choicePMT==a)))[0]  
            model.p_minusDelta_true[a] = len(idx)/(model.env.n_pmt/2)

        model.df_p1 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),\
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_plusDelta,
            'type':                 ['model']*12,
            'pmt-type':             ['+']*12 })

        model.df_p2 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),\
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_plusDelta_true,
            'type':                 ['data']*12,
            'pmt-type':             ['+']*12 })

        model.df_p3 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),\
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_minusDelta,
            'type':                 ['model']*12,
            'pmt-type':             ['-']*12 })

        model.df_p4 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),\
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_minusDelta_true,
            'type':                 ['data']*12,
            'pmt-type':             ['-']*12 })

        model.df_p = pd.concat([model.df_p1, model.df_p2, model.df_p3, model.df_p4], \
                ignore_index=True, sort=False)
        
        new_models.append(model)

    # Plotting
    for model in new_models:
        if 'df_p' not in locals():
            df_p = model.df_p
            df_p['model'] = model.model
        else:
            df_p_new = model.df_p
            df_p_new['model'] = model.model
            df_p = pd.concat( [df_p, df_p_new], \
                ignore_index=True, sort=False)

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p[df_p['model']==modelType], x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.png')
        plt.close()

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p[(df_p['model']==modelType) & (df_p['pmt-type']=='+')], 
            x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}+Delta.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}+Delta.png')
        plt.close()

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p[(df_p['model']==modelType) & (df_p['pmt-type']=='-')], 
            x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}-Delta.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}-Delta.png')
        plt.close()

    del df_p, df_p_new, df


# Cycle through directories
pathlist = Path(f'./figures/rangelTask/').glob('*')
modelTypes = ['dra', 'equalPrecision', 'freqBased']

for path in pathlist:
    # load models and dfs
    models = pickle.load(open(f'{str(path)}/models','rb'))
    df = pickle.load(open(f'{str(path)}/df','rb'))

    # # plot and save sigmas
    # sns.lineplot(data=df, x='state', y='sigma', hue='model')
    # plt.xticks([0, 1, 2, 3], ['s1', 's2', 's3', 's4'])
    # plt.ylim([1.5, 5.2])
    # plt.savefig(f'{str(path)}/sigma.svg')
    # plt.savefig(f'{str(path)}/sigma.png')
    # plt.close()

    # Compute probability that the model picks deterministic option
    new_models = []
    for model in models:
        model.p_plusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])
        model.p_minusDelta = norm.cdf(model.env.delta_pmt / model.sigma[:12])

        model.p_plusDelta_true = np.zeros(12)
        model.p_minusDelta_true = np.zeros(12)
        for a in range(12):
            # +Delta
            idx = (np.where((model.env.pmt_trial>0) & (model.choicePMT==a)))[0] 
            model.p_plusDelta_true[a] = len(idx)/(model.env.n_pmt/2)
            
            # -Delta
            idx = (np.where((model.env.pmt_trial<0) & (model.choicePMT==a)))[0]  
            model.p_minusDelta_true[a] = len(idx)/(model.env.n_pmt/2)

        # Plotting p(choice) to compare models
        model.df_p1 = pd.DataFrame( {
                'state':                np.repeat(['s1','s2','s3','s4'],3),       
                'action':               model.env.actions[:12],
                'p(choose better option)':model.p_plusDelta,
                'type':                 ['model']*12,
                'pmt-type':             ['+']*12 })

        # compute model.df_p1_diff 
        stateDiff, actions, pChooseBetterDiff = [], [], []
        for (si,sj) in [('s1','s2'),('s3','s4'),('s1','s3'),('s2','s4')]:
            for a in [0,1,2]:
                stateDiff.append(f'{si}-{sj}')
                actions.append(a)
                p_si_a = model.df_p1.loc [ (model.df_p1['state']==si) & (model.df_p1['action']%3==a) , 'p(choose better option)'].iloc[0]
                p_sj_a = model.df_p1.loc [ (model.df_p1['state']==sj) & (model.df_p1['action']%3==a) , 'p(choose better option)'].iloc[0]
                pChooseBetterDiff.append(p_si_a - p_sj_a)

        model.df_p1_diff = pd.DataFrame( {
                'state':                stateDiff,
                'action':               actions,
                'p(choose better option)':pChooseBetterDiff,
                'type':                 ['model']*len(actions),
                'pmt-type':             ['+']*len(actions) })


        ##############################
        model.df_p2 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_plusDelta_true,
            'type':                 ['data']*12,
            'pmt-type':             ['+']*12 })

        # compute model.df_p1_diff 
        stateDiff, actions, pChooseBetterDiff = [], [], []
        for (si,sj) in [('s1','s2'),('s3','s4'),('s1','s3'),('s2','s4')]:
            for a in [0,1,2]:
                stateDiff.append(f'{si}-{sj}')
                actions.append(a)
                p_si_a = model.df_p2.loc [ (model.df_p2['state']==si) & (model.df_p2['action']%3==a) , 'p(choose better option)'].iloc[0]
                p_sj_a = model.df_p2.loc [ (model.df_p2['state']==sj) & (model.df_p2['action']%3==a) , 'p(choose better option)'].iloc[0]
                pChooseBetterDiff.append(p_si_a - p_sj_a)

        model.df_p2_diff = pd.DataFrame( {
                'state':                stateDiff,
                'action':               actions,
                'p(choose better option)':pChooseBetterDiff,
                'type':                 ['data']*len(actions),
                'pmt-type':             ['+']*len(actions) })

        #############################
        model.df_p3 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_minusDelta,
            'type':                 ['model']*12,
            'pmt-type':             ['-']*12 })

        # compute model.df_p1_diff 
        stateDiff, actions, pChooseBetterDiff = [], [], []
        for (si,sj) in [('s1','s2'),('s3','s4'),('s1','s3'),('s2','s4')]:
            for a in [0,1,2]:
                stateDiff.append(f'{si}-{sj}')
                actions.append(a)
                p_si_a = model.df_p3.loc [ (model.df_p3['state']==si) & (model.df_p3['action']%3==a) , 'p(choose better option)'].iloc[0]
                p_sj_a = model.df_p3.loc [ (model.df_p3['state']==sj) & (model.df_p3['action']%3==a) , 'p(choose better option)'].iloc[0]
                pChooseBetterDiff.append(p_si_a - p_sj_a)

        model.df_p3_diff = pd.DataFrame( {
                'state':                stateDiff,
                'action':               actions,
                'p(choose better option)':pChooseBetterDiff,
                'type':                 ['model']*len(actions),
                'pmt-type':             ['+']*len(actions) })

        #############################
        model.df_p4 = pd.DataFrame( {
            'state':                np.repeat(['s1','s2','s3','s4'],3),
            'action':               model.env.actions[:12],
            'p(choose better option)':model.p_minusDelta_true,
            'type':                 ['data']*12,
            'pmt-type':             ['-']*12 })

        # compute model.df_p1_diff 
        stateDiff, actions, pChooseBetterDiff = [], [], []
        for (si,sj) in [('s1','s2'),('s3','s4'),('s1','s3'),('s2','s4')]:
            for a in [0,1,2]:
                stateDiff.append(f'{si}-{sj}')
                actions.append(a)
                p_si_a = model.df_p4.loc [ (model.df_p4['state']==si) & (model.df_p4['action']%3==a) , 'p(choose better option)'].iloc[0]
                p_sj_a = model.df_p4.loc [ (model.df_p4['state']==sj) & (model.df_p4['action']%3==a) , 'p(choose better option)'].iloc[0]
                pChooseBetterDiff.append(p_si_a - p_sj_a)

        model.df_p4_diff = pd.DataFrame( {
                'state':                stateDiff,
                'action':               actions,
                'p(choose better option)':pChooseBetterDiff,
                'type':                 ['data']*len(actions),
                'pmt-type':             ['+']*len(actions) })


        model.df_p = pd.concat([model.df_p1, model.df_p2, model.df_p3, model.df_p4], \
                ignore_index=True, sort=False)

        model.df_p_diff = pd.concat([model.df_p1_diff, model.df_p2_diff, model.df_p3_diff, model.df_p4_diff], \
                ignore_index=True, sort=False)
        
        new_models.append(model)

    # Plotting
    for model in new_models:
        if 'df_p' not in locals():
            df_p = model.df_p
            df_p['model'] = model.model
        else:
            df_p_new = model.df_p
            df_p_new['model'] = model.model
            df_p = pd.concat( [df_p, df_p_new], \
                ignore_index=True, sort=False)

    # Plotting
        for model in new_models:
            if 'df_p_diff' not in locals():
                df_p_diff = model.df_p_diff
                df_p_diff['model'] = model.model
            else:
                df_p_new_diff = model.df_p_diff
                df_p_new_diff['model'] = model.model
                df_p_diff = pd.concat( [df_p_diff, df_p_new_diff], \
                    ignore_index=True, sort=False)

    # for modelType in modelTypes:
    #     f,ax=plt.subplots()
    #     sns.violinplot(data=df_p[df_p['model']==modelType], x='state', y='p(choose better option)', hue='type')
    #     plt.title(f'{modelType}')
    #     plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.svg')
    #     plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.png')
    #     plt.close()

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p_diff[df_p_diff['model']==modelType], x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.show()
        # plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.svg')
        # plt.savefig(f'{str(path)}/p_choicePMT_{modelType}.png')
        # plt.close()

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p_diff[(df_p_diff['model']==modelType) & (df_p_diff['pmt-type']=='+')], 
            x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}+Delta.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}+Delta.png')
        plt.close()

    for modelType in modelTypes:
        f,ax=plt.subplots()
        sns.boxplot(data=df_p_diff[(df_p_diff['model']==modelType) & (df_p_diff['pmt-type']=='-')], 
            x='state', y='p(choose better option)', hue='type')
        plt.title(f'{modelType}')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}-Delta.svg')
        plt.savefig(f'{str(path)}/p_choicePMT_{modelType}-Delta.png')
        plt.close()





        

    # # plot, compute, and save sigma est. vs true
    # new_models = []
    # for model in models: 
    #     model.sigma_pmt1 = np.zeros(len(model.sigma)) 
    #     model.sigma_pmt2 = np.zeros(len(model.sigma)) 
    #     for a in range(12):
    #         # +Delta 
    #         idx = (np.where((model.env.pmt_trial>0) & (model.choicePMT==a)))[0] 
    #         p = np.clip(1 - len(idx)/10, 0.05, 0.95)    # n_pmt = 20
    #         model.sigma_pmt1[a] = model.env.delta_pmt/norm.ppf(p)
            
    #         # -Delta
    #         idx = (np.where((model.env.pmt_trial<0) & (model.choicePMT==a)))[0]  
    #         p = np.clip(1 - len(idx)/10, 0.05, 0.95)    # n_pmt = 20
    #         model.sigma_pmt2[a] = model.env.delta_pmt/norm.ppf(1-p)

    #         model.sigma_pmt = (model.sigma_pmt1 + model.sigma_pmt2)/2
        
    #     model.df_sigEst = pd.DataFrame( {
    #         'state':        np.repeat(np.arange(4),3),
    #         'action':       model.env.actions[:12],
    #         'q':            np.around(model.q[:12],1),
    #         'sigma':        np.around(model.sigma[:12],1),
    #         'sigma_est':    np.around(model.sigma_pmt[:12],1) })

    #     new_models.append(model)

    # fh = open(f'{savePath}new_models','wb')
    # pickle.dump(results, fh, pickle.HIGHEST_PROTOCOL)
    # fh.close()