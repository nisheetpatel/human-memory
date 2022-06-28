import os
import fnmatch
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# extract lmda, sigmaBase, delta_pmt, p-val for each expt
def extract_params(expt):
    
    lmdas = [model.lmda for model in expt.models[:10]]
    sigBs = [model.sigmaBase for model in expt.models[:10]]
    dltas = [model.env.delta_pmt for model in expt.models[:10]]
    
    # extract df to compute p-value metric per subject
    df = expt.df[expt.df['model']=='dra'].copy()
    df['subject'] = np.repeat(np.arange(10),8)

    # initialize arrays to collect df columns
    modelComparison, pvals, sub_id = [], [], []

    for subject in np.arange(10):
        # compute p-value per subject
        for (si,sj) in [('s1','s2'),('s3','s4'), ('s1','s3'),('s2','s4')]:
            
            # compute t-statistic for state combos
            p_si = df.loc [ (df['state']==si) & (df['subject']==subject), 'p(choose better option)'] 
            p_sj = df.loc [ (df['state']==sj) & (df['subject']==subject), 'p(choose better option)']
            result = stats.ttest_rel(p_si, p_sj)

            # define state combos
            if (si,sj) in [('s1','s2'),('s3','s4')]:
                modelComp = 'freq'
            elif (si,sj) in [('s1','s3'),('s2','s4')]:
                modelComp = 'stakes'

            # define columns of df with results
            modelComparison += [modelComp]
            pvals  += [result[1]]
            sub_id += [subject]

        # construct stats df
        df_stats = pd.DataFrame({
            'Model comp.': modelComparison,
            'p-value':     pvals,
            'subject':     sub_id
        })

    # min-max to get final p-value metric
    df_pvals_specific = df_stats.groupby(['Model comp.','subject'])\
                                .agg({'p-value': 'min'}).reset_index()

    df_pvals = df_pvals_specific.groupby(['subject'])\
                                .agg({'p-value': 'max'}).reset_index()

    # p-value metric column
    pvals = list(df_pvals['p-value'])

    
    # output dataframe
    df_params = pd.DataFrame({
        'lmda':         lmdas,
        'sigmaBase':    sigBs,
        'delta_pmt':    dltas,
        'p-value':      pvals,
        'log-p':      -np.log(pvals)
    })

    return df_params



# define mega df to collect all extracted params
df_params_all = pd.DataFrame({
    'lmda':     [],
    'sigmaBase':[],
    'delta_pmt':[],
    'p-value':  [],
    'log-p':    []
})


# define path for files
rootdir = f'./figures/'


# cycle through directories then load relevant files
for subdir, dirs, files in os.walk(rootdir):
    
    print(f'\n\nProcessing {subdir}')

    for file in files:

        if not fnmatch.fnmatch(file, '*.*g'):
            
            print(f'\t{file}')

            # load file
            fh = open(os.path.join(subdir, file), 'rb')
            exptGroup = pickle.load(fh)
            fh.close()

            # get params from each experiment
            for idx, expt in enumerate(exptGroup.expts[0][50:]): 
                
                df_params = extract_params(expt)
                df_params_all = pd.concat([df_params_all, df_params],
                            ignore_index=True, sort=False)


# 4D plots with plotly
import plotly
import plotly.graph_objs as go

#Read cars data from csv
data = df_params_all

#Set marker properties
markercolor = data['p-value']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['lmda'],
                    y=data['sigmaBase'],
                    z=data['delta_pmt'],
                    marker=dict(color=markercolor,
                                opacity=1,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="lmda"),
                                yaxis=dict( title="sigmaBase"),
                                zaxis=dict(title="delta_pmt")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("4DPlot.html"))


from matplotlib.colors import LogNorm

# Plotting with seaborn 

for delta_pmt in [1,2,3,4]:

    df = data[data['delta_pmt']==delta_pmt].copy()
    df = df.drop(['delta_pmt'], axis=1)

    # pval = np.array(df['p-value'].copy())
    
    # pval_new = pval.copy()
    # for thresh in [[0,0.00001],[0.00001,0.0001],[0.0001,0.001],[0.001,0.01],[0.01,0.05],[0.05,0.5],[0.5,1]]: 
    #     pval_new[(pval>=thresh[0]) & (pval<thresh[1])] = thresh[1]

    # df['p-value'] = pval_new
    # df['p-value'] = df['p-value'].astype('category')

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='lmda', y='sigmaBase', hue='log-p', palette='rocket', ax=ax)

    ax.set_title('DRA: $\Delta_{PMT}$ ='+f'{delta_pmt}')
    ax.set_ylim([0,20])
    ax.set_xlim([0,0.75])
    plt.savefig(f'./figures/scatter_delta{delta_pmt}.png')
    plt.close()