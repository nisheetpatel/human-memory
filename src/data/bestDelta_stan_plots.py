# stan code
from stan_code import prob_of_signature
import numpy as np
from numpy.core.shape_base import block
import pandas as pd
from extractFakeDataStuff import extract_posteriors_and_params
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# fetch posteriors and corresponding experiment params to analyze
modelTypes  = ['dra','freq-s','stakes','equalPrecision']
lmda        = 0.1
sigmaBase   = 10
posteriors, df_expt_params = extract_posteriors_and_params(lmda=lmda, sigmaBase=sigmaBase)

# for expt_id, posterior in enumerate(posteriors):
#     if
# posterior is not None:
expt_id_chunks = []
for delta_pmt in df_expt_params['delta_pmt'].unique():
    expt_id_chunks += [df_expt_params.loc[
        (df_expt_params['lmda']==lmda) & 
        (df_expt_params['sigmaBase']==sigmaBase) &
        (df_expt_params['delta_pmt']==delta_pmt),
        'expt_id'].to_list()]

for expt_ids in expt_id_chunks:
    f,axs = plt.subplots(5)
    f.set_figheight(12)
    f.set_figwidth(4.5)

    for expt_id, ax in zip(expt_ids,axs):

            posterior = posteriors[expt_id]
            delta_pmt = df_expt_params.loc[expt_id, "delta_pmt"]

            models, states, cas = [], [], []
            l = len(posterior[0].flatten())

            for i in range(len(modelTypes)):
                cas    += list(posterior[i].flatten())
                models += [modelTypes[i]] * l
                repL    = int(l/len(modelTypes))
                states += list(np.repeat(['s1','s2','s3','s4'], repL))

            dfPlot = pd.DataFrame({
                'Model':            models,
                'State':            states,
                'ChoiceAccuracy':   cas
                })

            _ = sns.lineplot(data=dfPlot, x='State', y='ChoiceAccuracy', 
                        hue='Model', style='Model', ci='sd', ax=ax)
            ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels)
    for ax in axs:
        ax.get_legend().remove()
    t = f'$\lambda = {lmda}, \sigma_b = {sigmaBase}, \Delta={delta_pmt}$'
    axs[0].set_title(t)
    f.savefig(f'./Plots/posteriors_varyParams/ChoiceAccuracy_lmda{int(lmda*100)}_sb_{int(sigmaBase)}_deltaPMT_{int(delta_pmt)}p{str(delta_pmt%1)[-1]}_5runs.svg')
    plt.close()


# tests
# fetch posteriors and corresponding experiment params to analyze
dfPlot_list = []

modelTypes  = ['dra','freq-s','stakes','equalPrecision']

for lmda, sigmaBase in itertools.product(lmdas,sigmaBases):
    
    # sigmaBase   = 2.5
    posteriors, df_expt_params = extract_posteriors_and_params(lmda=lmda, sigmaBase=sigmaBase)

    expt_id_chunks = []
    for delta_pmt in df_expt_params['delta_pmt'].unique():
        expt_id_chunks += [df_expt_params.loc[
            (df_expt_params['lmda']==lmda) & 
            (df_expt_params['sigmaBase']==sigmaBase) &
            (df_expt_params['delta_pmt']==delta_pmt),
            'expt_id'].to_list()]

    mats = []
    mat_acc = []

    for expt_ids in expt_id_chunks:

        for expt_id in expt_ids:

                # extract posterior
                posterior = posteriors[expt_id]
                delta_pmt = df_expt_params.loc[expt_id, "delta_pmt"]
                
                # get prob. of model signature
                prob_sig = prob_of_signature(posterior)
                mats += [prob_sig]
                mat_acc += [np.trace(prob_sig)/np.sum(prob_sig)]

        # # plot all 5 matrices
        # mat = np.mean(mats[-len(expt_ids):], axis=0)
        # f,ax = plt.subplots()
        # ax = plt.imshow(mat)
        # plt.colorbar()
        # t = f'$\lambda = {lmda}, \sigma_b = {sigmaBase}, \Delta={delta_pmt}$'
        # plt.title(t)
        # plt.xticks(list(range(len(modelTypes))), modelTypes)
        # plt.yticks(list(range(len(modelTypes))), modelTypes)
        # f.savefig(f'./Plots/posteriors_varyParams/ModelPostProbMatrix_lmda{int(lmda*100)}_sb_{int(sigmaBase)}_deltaPMT_{int(delta_pmt)}p{str(delta_pmt%1)[-1]}_5runs.svg')
        # plt.close()

    df_expt_params['RecoveryMetric'] = mat_acc

    dfPlot_list += [df_expt_params]

dfPlot = pd.concat(dfPlot_list, sort=False, ignore_index=True)
sns.lineplot(data=dfPlot, x='delta_pmt', y='RecoveryMetric', hue='lmda')
plt.title(f'$\sigma_b=${sigmaBase}')
plt.savefig(f'./Plots/posteriors_varyParams/ModelRecovery_sb_{int(sigmaBase)}.svg')
plt.close()





##############################################################

# tests
# fetch posteriors and corresponding experiment params to analyze
dfPlot_list = []

modelTypes  = ['dra','freq-s','stakes','equalPrecision']

for lmda, sigmaBase in itertools.product(lmdas,sigmaBases):
    
    df_expt_params = df_full.loc[(df_full['lmda']==lmda) & (df_full['sigmaBase']==sigmaBase)]

    expt_id_chunks = []
    for delta_pmt in df_expt_params['delta_pmt'].unique():
        expt_id_chunks += [df_expt_params.loc[
            (df_expt_params['lmda']==lmda) & 
            (df_expt_params['sigmaBase']==sigmaBase) &
            (df_expt_params['delta_pmt']==delta_pmt),
            'expt_id'].to_list()]

    mats = []
    mat_acc = []

    for expt_ids in expt_id_chunks:

        for expt_id in expt_ids:

            if posteriors[expt_id] is not None:
                # extract posterior
                posterior = posteriors[expt_id]
                delta_pmt = df_expt_params.loc[expt_id, "delta_pmt"]
                
                # get prob. of model signature
                prob_sig = prob_of_signature(posterior)
                mats += [prob_sig]
                mat_acc += [np.trace(prob_sig)/np.sum(prob_sig)]

            else:
                mats += [np.nan]
                mat_acc += [np.nan]

        # # plot all 5 matrices
        # mat = np.mean(mats[-len(expt_ids):], axis=0)
        # f,ax = plt.subplots()
        # ax = plt.imshow(mat)
        # plt.colorbar()
        # t = f'$\lambda = {lmda}, \sigma_b = {sigmaBase}, \Delta={delta_pmt}$'
        # plt.title(t)
        # plt.xticks(list(range(len(modelTypes))), modelTypes)
        # plt.yticks(list(range(len(modelTypes))), modelTypes)
        # f.savefig(f'./Plots/posteriors_varyParams/ModelPostProbMatrix_lmda{int(lmda*100)}_sb_{int(sigmaBase)}_deltaPMT_{int(delta_pmt)}p{str(delta_pmt%1)[-1]}_5runs.svg')
        # plt.close()

    if len(mat_acc)>0:
        df_expt_params['RecoveryMetric'] = mat_acc

        dfPlot_list += [df_expt_params]

dfPlot = pd.concat(dfPlot_list, sort=False, ignore_index=True)
sns.lineplot(data=dfPlot, x='delta_pmt', y='RecoveryMetric', hue='lmda')
plt.title(f'$\sigma_b=${sigmaBase}')
plt.savefig(f'./Plots/posteriors_varyParams/ModelRecovery_sb_{int(sigmaBase)}.svg')
plt.close()


##########
# Plotting to check range of choice accuracies
def extract_posteriors(lmda=0.1, sigmaBase=10):
    # get indices of experiments to analyze
    df = pd.read_pickle('./simulatedData/exptParamsDF_varyParams')
    ids = df.loc[(df['lmda']==lmda) & (df['sigmaBase']==sigmaBase), 'expt_id'].to_list()
    df_expt_params = df.iloc[ids]

    # load the corresponding posteriors
    posteriors = [None] * (len(df))
    for id in ids:
        posteriors[id] = np.load(f'./simulatedData/posteriors_varyParams/mu_ca_{id}.npy')

    return posteriors, df_expt_params


# fetch posteriors and corresponding experiment params to analyze
dfPlot_list = []

modelTypes  = ['dra','freq-s','stakes','equalPrecision']

for lmda, sigmaBase in itertools.product(lmdas,sigmaBases):

    # sigmaBase   = 2.5
    posteriors, df_expt_params = extract_posteriors(lmda=lmda, sigmaBase=sigmaBase)

    expt_id_chunks = []
    for delta_pmt in df_expt_params['delta_pmt'].unique():
        expt_id_chunks += [df_expt_params.loc[
            (df_expt_params['lmda']==lmda) &
            (df_expt_params['sigmaBase']==sigmaBase) &
            (df_expt_params['delta_pmt']==delta_pmt),
            'expt_id'].to_list()]

    mean_ca = []
    ca_s1 = []
    ca_s2 = []
    ca_s3 = []
    ca_s4 = []
    dra_ca = []
    ep_ca = []
    freq_ca = []
    stakes_ca = []
    dra_ca_s1 = []
    dra_ca_s2 = []
    dra_ca_s3 = []
    dra_ca_s4 = []
    mats = []
    mat_acc = []

    for expt_ids in expt_id_chunks:

        for expt_id in expt_ids:

                # extract posterior
                posterior = posteriors[expt_id]
                delta_pmt = df_expt_params.loc[expt_id, "delta_pmt"]

                # get prob. of model signature
                prob_sig = prob_of_signature(posterior)
                mats += [prob_sig]
                mat_acc += [np.trace(prob_sig)/np.sum(prob_sig)]

                # get mean choice accuracy
                mean_ca += [posterior.mean()]
                ca_s1   += [posterior[:,0].mean()]
                ca_s2   += [posterior[:,1].mean()]
                ca_s3   += [posterior[:,2].mean()]
                ca_s4   += [posterior[:,3].mean()]
                dra_ca  += [posterior[0].mean()]
                freq_ca += [posterior[1].mean()]
                stakes_ca+=[posterior[2].mean()]
                ep_ca  +=  [posterior[3].mean()]
                dra_ca_s1 += [posterior[0,0].mean()]
                dra_ca_s2 += [posterior[0,1].mean()]
                dra_ca_s3 += [posterior[0,2].mean()]
                dra_ca_s4 += [posterior[0,3].mean()]

    df_expt_params['RecoveryMetric'] = mat_acc
    df_expt_params['ChoiceAccuracy'] = mean_ca
    df_expt_params['ChoiceAccuracy-s1'] = ca_s1
    df_expt_params['ChoiceAccuracy-s2'] = ca_s2
    df_expt_params['ChoiceAccuracy-s3'] = ca_s3
    df_expt_params['ChoiceAccuracy-s4'] = ca_s4
    df_expt_params['ChoiceAccuracy-DRA'] = dra_ca
    df_expt_params['ChoiceAccuracy-Freq'] = freq_ca
    df_expt_params['ChoiceAccuracy-Stakes'] = stakes_ca
    df_expt_params['ChoiceAccuracy-EP'] = ep_ca
    df_expt_params['ChoiceAccuracy-DRA-s1'] = dra_ca_s1
    df_expt_params['ChoiceAccuracy-DRA-s2'] = dra_ca_s2
    df_expt_params['ChoiceAccuracy-DRA-s3'] = dra_ca_s3
    df_expt_params['ChoiceAccuracy-DRA-s4'] = dra_ca_s4

    dfPlot_list += [df_expt_params]


dfPlot = pd.concat(dfPlot_list, sort=False, ignore_index=True)
dfPlot['Delta'] = dfPlot['delta_pmt']/dfPlot['sigmaBase']


toPlot = ['ChoiceAccuracy',
'ChoiceAccuracy-DRA',
'ChoiceAccuracy-Freq',
'ChoiceAccuracy-Stakes',
'ChoiceAccuracy-EP',
'ChoiceAccuracy-DRA-s1',
'ChoiceAccuracy-DRA-s2',
'ChoiceAccuracy-DRA-s3',
'ChoiceAccuracy-DRA-s4']

for itemToPlot in toPlot:
    sns.lineplot(data=dfPlot, x=itemToPlot, y='RecoveryMetric', hue='lmda')
    plt.savefig(f'./Plots/posteriors_varyParams/ModelRecovery_{itemToPlot}.svg')
    plt.close()