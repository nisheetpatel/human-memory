import os
import fnmatch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# add 10% lapses to data
def add_lapses(df, N=60, lapse_factor=0.1):
    p_col = np.array(df['p(choose better option)'])

    # define len(p_col)xN array with randomly shuffled choices
    arr = np.zeros((len(p_col),N))
    for idx, _ in enumerate(list(arr)):
        arr[idx,:int(p_col[idx]*N)] = 1

    # coin-flip N*lapse_factor random elements 
    for row in arr:
        np.random.shuffle(row)
    arr[:,-int(N*lapse_factor):] = np.random.choice([0,1],\
        size=(len(p_col), int(N*lapse_factor)))

    # adding the new p_col
    df_new = df.copy()
    df_new['p(choose better option)'] = np.sum(arr, axis=1)/N

    return df_new


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

            # mark no lapses in ones already present 
            exptGroup.df_pvals['lapses'] = False
            exptGroup.df_stats['lapses'] = False

            # extract df of interest and add lapses
            for idx, expt in enumerate(exptGroup.expts[0][50:]): 
                
                # do many times?
                # add 10% lapses
                expt.df = add_lapses(expt.df)
                expt.df_stats, expt.df_pvals = expt.computeStats(expt_id=idx%50, 
                                            varySubjectParams=bool(idx>=50))
                
                # add column to mark lapses
                expt.df_stats['lapses'] = True
                expt.df_pvals['lapses'] = True

                # concat df for all experiments for final plots
                # collect results across experiments in dfs
                exptGroup.df_pvals = pd.concat([exptGroup.df_pvals, expt.df_pvals],
                                ignore_index=True, sort=False)

                exptGroup.df_stats = pd.concat([exptGroup.df_stats, expt.df_stats],
                                ignore_index=True, sort=False)

            # Plot and save effect of lapses
            fig, ax = plt.subplots()

            df = exptGroup.df_pvals[(exptGroup.df_pvals['Model']=='dra') &\
                (exptGroup.df_pvals['Test']=='test') &\
                (exptGroup.df_pvals['vary subject params']==False) ]

            sns.histplot(df, x='p-value', hue='lapses', \
                log_scale=True, bins=20, ax=ax, stat='density')

            ax.set_title('DRA: $\Delta_{PMT}$ ='+f'{exptGroup.params["delta_pmt"]}')
            ax.axvline(0.05,linestyle='--',color='gray') 
            
            plt.savefig(os.path.join(subdir, file)+f'_lapses20.svg')
            plt.close()