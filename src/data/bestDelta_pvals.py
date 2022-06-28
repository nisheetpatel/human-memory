import itertools
import numpy as np
import pandas as pd
from experiment import Experiment

# Defining parameter set to train on
lmdas 		= np.arange(0.05,0.25,0.05)
sigmaBases 	= np.arange(2,16.1,2)
delta_pmts 	= np.arange(2,5.1,0.5)
delta_1s 	= [4]
delta_2s 	= [1]

def trainAllParams(lmdas, sigmaBases, delta_pmts, delta_1s, delta_2s):

    # Dataframe to collect stuff
    df = pd.DataFrame({
        'lmda':     [],
        'sigmaBase':[],
        'p-value':  [],
        '-log(p)':  []
    })

    for params in itertools.product(lmdas, sigmaBases,
                delta_pmts, delta_1s, delta_2s):

        # Defining the dictionary to pass
        paramsDict = {
            'lmda':             params[0],
            'sigmaBase':        params[1],
            'delta_pmt':        params[2],
            'delta_1':          params[3],
            'delta_2':          params[4]
            }
        
        # Printing to keep track
        print(f'\nRunning param set:')
        print(paramsDict)

        # Running experiment
        expt = Experiment(**paramsDict)
        expt.run() 
        expt.gatherData() 
        expt.df_stats, expt.df_pvals = expt.computeStats(varySubjectParams=False)

        df_expt = pd.DataFrame({
            'lmda':     paramsDict['lmda'],
            'sigmaBase':paramsDict['sigmaBase'],
            'delta_pmt':paramsDict['delta_pmt'],
            'p-value':  expt.df_pvals.loc[[0]]['p-value'],
            '-log(p)':  -np.log10(expt.df_pvals.loc[[0]]['p-value'])
        })
        df = pd.concat([df,df_expt], sort=False, ignore_index=True)

    return df


if __name__ == '__main__':
    # Defining parameter set to train on
    paramSet = {
        'lmdas': 		np.arange(0, 0.51, 0.025),
        'sigmaBases': 	np.arange(2, 20.1, 2),
        'delta_pmts': 	np.repeat(np.array([2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7]), 5),
        'delta_1s': 	[4],
        'delta_2s': 	[1]
    }
    
    df = trainAllParams(**paramSet)

    import pickle
    fh = open('./figures/df_pvals_deltaPMT_new','wb')
    pickle.dump(df, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()


# #######################################
# # Plotting
# #######################################
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D

# dfPlot = df.copy()

# # Get the mean (log) p-values (mean across multiple experiments of same param values)
# dfPlot = dfPlot.groupby(['lmda', 'sigmaBase', 'delta_pmt'])\
#                 .agg({'-log(p)': 'mean'})\
#                 .reset_index()

# # define color palette and add corresponding column to dataframe
# palette = sns.color_palette("husl", len(dfPlot['delta_pmt'].unique()))
# color_dict = {'2.0': palette[0], '3.0': palette[1], '3.5': palette[2], 
#               '4.0': palette[3], '4.5': palette[4], '5.0': palette[5], 
#               '5.5': palette[6], '6.0': palette[7], '7.0': palette[8]} 
# dfPlot['color'] = dfPlot['delta_pmt'].apply( lambda x: color_dict[str(float(x))] )  


# fig = plt.figure() 
# ax = fig.add_subplot(111, projection='3d')

# for delta_pmt in [2,3,3.5,5.5,5,4.5,4]: #np.sort(dfPlot['delta_pmt'].unique()):
#     df1 = dfPlot[dfPlot['delta_pmt']==delta_pmt]
#     # ax.scatter(df1['lmda'], df1['sigmaBase'], df1['-log(p)'], c=color, label=delta_pmt)
#     surf = ax.plot_trisurf(df1['lmda'], df1['sigmaBase'], df1['-log(p)'], label=delta_pmt, alpha=0.5)
#     surf._facecolors2d=surf._facecolor3d
#     surf._edgecolors2d=surf._edgecolor3d

# # axis labels and title
# ax.set_xlabel('$\lambda$')
# ax.set_ylabel('$\sigma_{base}$')
# ax.set_zlabel('p-value exponent')

# ax.view_init(30, 185) 
# ax.legend()
# plt.show()



# #############################################
# # plot max Delta-PMT
# #############################################

# dfPlot = df.copy()

# # Get the mean (log) p-values (mean across multiple experiments of same param values)
# dfPlot = dfPlot.groupby(['lmda', 'sigmaBase', 'delta_pmt'])\
#                 .agg({'-log(p)': 'mean'})\
#                 .reset_index()

# # Keep only the p-value that corresponds to the max Delta_PMT
# idx = dfPlot.groupby(['lmda', 'sigmaBase'], sort=False)['-log(p)'].transform(max) == dfPlot['-log(p)']
# dfPlot = dfPlot[idx]

# # define color palette and add corresponding column to dataframe
# palette = sns.color_palette("husl", len(dfPlot['delta_pmt'].unique())) 
# color_dict = {'2.0': palette[0], '3.0': palette[1], '3.5': palette[2], 
#               '4.0': palette[3], '4.5': palette[4], '5.0': palette[5], 
#               '5.5': palette[6], '6.0': palette[7], '7.0': palette[8]} 
# dfPlot['color'] = dfPlot['delta_pmt'].apply( lambda x: color_dict[str(float(x))] )  

# # plot 
# fig, ax = plt.subplots()

# # produce color legend
# for delta_pmt in np.sort(dfPlot['delta_pmt'].unique()):
#     # slice dataframe
#     df1 = dfPlot[dfPlot['delta_pmt']==delta_pmt].copy()
#     ax.scatter(df1['lmda'], df1['sigmaBase'], c=df1['color'], s=50, label=delta_pmt)
# legend1 = ax.legend(title="$\Delta_{PMT}$", loc='lower right')
# ax.add_artist(legend1)

# scatter = ax.scatter(dfPlot['lmda'], dfPlot['sigmaBase'], cmap=palette,
#             c=dfPlot['color'], s=dfPlot['-log(p)']*20, label=dfPlot['delta_pmt'])

# # produce another legend with a cross section of sizes from the scatter
# handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, func=(lambda x: x/20 ) )
# legend2 = ax.legend(handles, labels, title="p-value \nexponent", bbox_to_anchor=(1.01, 1))

# # axis labels and title
# ax.set_xlabel('$\lambda$')
# ax.set_ylabel('$\sigma_{base}$')
# ax.set_title('Best $\Delta_{PMT}$ for each set of subject parameters')

# plt.show()



##########################################
# plot all Delta-PMTs separated by colour
##########################################
# dfPlot = df.copy()
# dfPlot = dfPlot.groupby(['lmda', 'sigmaBase', 'delta_pmt'])\
#                 .agg({'-log(p)': 'mean'})\
#                 .reset_index()

# # define color palette and add corresponding column to dataframe
# palette = sns.color_palette("husl", len(dfPlot['delta_pmt'].unique())) 
# color_dict = {'2.0': palette[0], '3.0': palette[1], '3.5': palette[2], 
#               '4.0': palette[3], '4.5': palette[4], '5.0': palette[5], 
#               '5.5': palette[6], '6.0': palette[7], '7.0': palette[8]} 
# dfPlot['color'] = dfPlot['delta_pmt'].apply( lambda x: color_dict[str(float(x))] )  

# fig, ax = plt.subplots()

# for delta_pmt in np.sort(dfPlot['delta_pmt'].unique()):
    
#     # slice dataframe
#     df1 = dfPlot[dfPlot['delta_pmt']==delta_pmt].copy()

#     # jitter points on x-axis
#     df1['lmda'] = df1['lmda'] + (delta_pmt - 3.5) * 0.005

#     scatter = ax.scatter(df1['lmda'], df1['sigmaBase'], marker='|',
#             c=df1['color'], s=df1['-log(p)']*40, label=delta_pmt)

# ax.legend(title="$\Delta_{PMT}$", bbox_to_anchor=(1.01, 1))

# # # produce another legend with a cross section of sizes from the scatter
# # handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
# # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

# plt.show() 