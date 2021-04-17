import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import multiprocessing as mp
from train import train_model, save_results


""" Figure better, cleaner way to do this"""
# Redefining parameters
# initializing hyperparameters
n_restarts  = 10


keyStates  = [5, 6, 16, 17] 
keyStates0 = [5, 16]
correctAct = [7, 9, 18, 20]
keyActions = [7, 8, 9, 10, 18, 19, 20, 21]


# Defining model types to train
modelTypes = ['dra', 'equalPrecision', 'freq 1',\
            'freq 0.97', 'freq 0.95', 'freq 0.9']


# Running in parallel
pool = mp.Pool()
results = pool.map(train_model, \
    np.repeat(modelTypes, n_restarts))

savePath = save_results(results)

# Load dataframe for plotting
df = pickle.load(open(f'{savePath}df_before','rb'))


# Plotting all things at steady state
dfPlot = df[df.Episode >= (np.max(df.Episode)-100)]
# df2 = df[df.Episode>=(episodes+extraEps-100)]

for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x="Model", y=toPlot)
    plt.savefig(f'{savePath}Perf_{toPlot}_steadyState_beforeSwitch')
    plt.close()

# Best SIGMA
fig, ax = plt.subplots()
ax = sns.boxplot(data=dfPlot, x='Model', y='Sigma',\
    hue='Action', palette='Paired')
plt.savefig(f'{savePath}SIGMA_beforeSwitch')
plt.close()

# Best P_mistake, P_visits
# Defining same palette as sigma
custom_palette = sns.color_palette('Paired',8)
custom_palette = [custom_palette[i] for i in [1,3,5,7]]

for toPlot in ['P_visit', 'P_mistake', 'Choice accuracy']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x='Model', y=toPlot,\
        hue='State', palette=custom_palette)
    plt.savefig(f'{savePath}Obs_{toPlot}_steadyState_beforeSwitch')
    plt.close()


# Lineplot: Alex's preferred version for comparison
sns.catplot(data=dfPlot, x='State', y='P_mistake',\
        hue='Model', kind='point', palette='colorblind',\
        linestyles=['solid','dotted','dashed'],\
        markers=['o','v','s'])
plt.ylim([0,0.5])
plt.savefig(f'{savePath}Obs_P_mistake_lines_beforeSwitch')
plt.show()


# Noisy memories (instead of sigma)
import scipy.stats as stats
import math
sns.set_style("white")

for modelType in modelTypes:
    q = [20, -20, 20, -5, 20, -5, 20, -20]
    sig = [df['Sigma']
            [(df['Model']==modelType) & (df['Action']==a)]
            .median()
            for a in keyActions]
    stateMap = {'0':5, '2':6, '4':16, '6':17}
    for i in [0,2,4,6]:
        x = np.linspace(-150, 150, 1000)
        plt.plot(x, stats.norm.pdf(x, q[i], sig[i]))
        plt.plot(x, stats.norm.pdf(x, q[i+1], sig[i+1]))
        plt.title(f'State = {stateMap[str(i)]}, Model = {modelType}')
        plt.show()


####################################################
#                                                  #
# Task manipulation: remove stochastic choice sets #
#                                                  #
####################################################
extraEps = 500
argsToPass = [(None,model,extraEps) for model in results]
results2 = pool.starmap(train_model, argsToPass)

# Plot probability of making mistakes after switching the environment
    # initialising columns for dataframe 
col_model, col_state, col_pmistake = [], [], []

correctAct.extend([1,12])   # from bottleneck states

for model in results2:
    
    # Defining model type
    modelType = model.model
    
    if model.model == 'freqBased':
        if model.decay == 1:
            model.decay = int(1)
        modelType = f'freq '+str(model.decay)

    for state in [0,11]:
        
        col_model.append(modelType)        
        col_state.append(state)

        # compute prob. of making mistake
        model.env.state = state
        mistakes = 0
        for _ in range(1000):
            a, _, _, _ = model.act(state)
            if a not in correctAct:
                mistakes += 1

        col_pmistake.append(mistakes/1000)

# creating dataframe to plot results
dfPlot2 = pd.DataFrame({
        'Model':            col_model,\
        'State':            col_state,\
        'P_mistake':        col_pmistake})

fig, ax = plt.subplots()
ax = sns.boxplot(data=dfPlot2, x='Model', y='P_mistake',\
    hue='State', palette=custom_palette)
plt.savefig(f'{savePath}Obs_P_mistake_bottleneckStates_afterSwitch')
plt.close()

"""
Older plotting stuff
"""
# # Plotting Objective, expected reward, cost
# for toPlot in ['Objective', 'Expected reward', 'Cost']:
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=df, x="Episode", y=toPlot,\
#         hue="Model", style="Model", dashes=False)
#     plt.savefig(f'{savePath}Perf_{toPlot}')
#     plt.close()

# for modelType in modelTypes: 
#     fig, ax = plt.subplots() 
#     sns.lineplot(x='Episode', y='Sigma', hue='Action',\
#         data=df[df.Model==modelType], legend='full',\
#         palette='Paired')
#     ax.set_ylim([0,50])
    
#     if modelType.split()[0] == 'freq':
#         saveName = 'freq' + str(int((float\
#             (modelType.split()[1])*100)))
#     else:
#         saveName = modelType.split()[0]
    
#     plt.savefig(f'{savePath}Sigma_{saveName}_evolution')
#     plt.close()

# for toPlot in ['P_visit', 'P_mistake', 'Choice accuracy']:
#     fig, ax = plt.subplots()
#     ax = sns.boxplot(data=dfPlot, x="State", y=toPlot, hue='Model')
#     ax.set_title('Before switch')
#     plt.savefig(f'{savePath}Obs_{toPlot}_steadyState_beforeSwitch')
#     plt.close()

# # Creating a column of actions: 0 and 1 for split plotting
# col_action = np.array(dfPlot['Action'])
# for a in keyActions:
#     if a in correctAct:
#         col_action[col_action == a] = 0
#     else:
#         col_action[col_action == a] = 1
# dfPlot['action'] = col_action

# for modelType in modelTypes:
    
#     dfModel = dfPlot[dfPlot.Model == modelType]

#     if modelType.split()[0] == 'freq':
#         saveName = 'freq' + str(int((float\
#             (modelType.split()[1])*100)))
#         palette = 'Set3'
#     else:
#         saveName = modelType.split()[0]
#         if modelType == 'dra':
#             palette = 'Set1'
#         else:
#             palette = 'Set2'

#     fig, ax = plt.subplots()
#     ax = sns.violinplot(data=dfModel, x='State', y='Sigma',\
#         hue='action', split=True, palette=palette)
#     ax.set_title(f'{modelType} - Before switch')
#     plt.savefig(f'{savePath}Sigma_{saveName}_steadyState_beforeSwitch')
#     plt.close()






# # Plotting after
# # Plotting all things at steady state
# dfPlot = df[df.Episode >= (1200)]
# # df2 = df[df.Episode>=(episodes+extraEps-100)]

# for toPlot in ['Objective', 'Expected reward', 'Cost']:
#     fig, ax = plt.subplots()
#     ax = sns.boxplot(data=dfPlot, x="Model", y=toPlot)
#     plt.savefig(f'{savePath}Perf_{toPlot}_steadyState_afterSwitch')
#     plt.close()

# for toPlot in ['P_visit', 'P_mistake', 'Choice accuracy']:
#     fig, ax = plt.subplots()
#     ax = sns.boxplot(data=dfPlot, x="State", y=toPlot, hue='Model')
#     ax.set_title('After switch')
#     plt.savefig(f'{savePath}Obs_{toPlot}_steadyState_afterSwitch')
#     plt.close()

# # Creating a column of actions: 0 and 1 for split plotting
# col_action = np.array(dfPlot['Action'])
# for a in keyActions:
#     if a in correctAct:
#         col_action[col_action == a] = 0
#     else:
#         col_action[col_action == a] = 1
# dfPlot['action'] = col_action

# for modelType in modelTypes:
    
#     dfModel = dfPlot[dfPlot.Model == modelType]

#     if modelType.split()[0] == 'freq':
#         saveName = 'freq' + str(int((float\
#             (modelType.split()[1])*100)))
#         palette = 'Set3'
#     else:
#         saveName = modelType.split()[0]
#         if modelType == 'dra':
#             palette = 'Set1'
#         else:
#             palette = 'Set2'

#     fig, ax = plt.subplots()
#     ax = sns.violinplot(data=dfModel, x='State', y='Sigma',\
#         hue='action', split=True, palette=palette)
#     ax.set_title(f'{modelType} - After switch')
#     plt.savefig(f'{savePath}Sigma_{saveName}_steadyState_afterSwitch')
#     plt.close()

# # Plotting Objective, expected reward, cost
# for toPlot in ['Objective', 'Expected reward', 'Cost']:
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=df, x="Episode", y=toPlot,\
#         hue="Model", style="Model", dashes=False)
#     plt.savefig(f'{savePath}Perf_{toPlot}_full')
#     plt.close()
# 
# for modelType in modelTypes: 
#     fig, ax = plt.subplots() 
#     sns.lineplot(x='Episode', y='Sigma', hue='Action',\
#         data=df[df.Model==modelType], legend='full',\
#         palette='Paired')
#     ax.set_ylim([0,50])
    
#     if modelType.split()[0] == 'freq':
#         saveName = 'freq' + str(int((float\
#             (modelType.split()[1])*100)))
#     else:
#         saveName = modelType.split()[0]
    
#     plt.savefig(f'{savePath}Sigma_{saveName}_evolution_full')
#     plt.close()

###########################################
# # Plotting evolution of observables
# for modelType in modelTypes:
    
#     # Time updates
#     start = time()

#     # Extracting model's data
#     dfModel = df[df['Model']==modelType]

#     if modelType.split()[0] == 'freq':
#         saveName = 'freq' + str(int((float\
#             (modelType.split()[1])*100)))
#     else:
#         saveName = modelType.split()[0]

#     # Plotting p visit
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=dfModel, x="Episode", \
#         y="P_visit", hue="State",\
#         legend='full', palette="Set2")
#     ax.set_title(f'{modelType}')
#     ax.set_ylim([0,1])
#     #plt.savefig(f'{savePath}pvisit_{saveName}')
#     #plt.show()

#     # Plotting sigma
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=dfModel, x="Episode", \
#         y='Sigma', hue='Action', ci=None,\
#         legend='full', palette="Paired")
#     ax.set_title(f'{modelType}')
#     ax.set_ylim([0,55])
#     #plt.savefig(f'{savePath}sigma_{saveName}')
#     #plt.show()

#     # Plotting MA of P(mistake)
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=dfModel, x="Episode", \
#         y="P_mistake", hue="State",\
#         legend='full', palette="Paired")
#     ax.set_title(f'{modelType}')
#     ax.set_ylim([0,0.65])
#     #plt.savefig(f'{savePath}Obs_Pmistake_{saveName}')
#     #plt.show()

#     # Plotting choice accuracy
#     fig, ax = plt.subplots()
#     ax = sns.lineplot(data=dfModel, x="Episode",\
#         y="Choice accuracy", hue="State",\
#         legend='full', palette="Paired")
#     ax.set_title(f'{modelType}')
#     ax.set_ylim([0.5,1])
#     #plt.savefig(f'{savePath}Obs_choiceAccuracy_{saveName}')
#     #plt.show()

#     # Printing time updates
#     print(f'Time to plot for {modelType} = '\
#      f'{str(datetime.timedelta(seconds=time()-start))}')

#plt.close('all')

# # Saving the dataframe and models
# import pickle

# fh = open(f'{savePath}df','wb')
# pickle.dump(df, fh, pickle.HIGHEST_PROTOCOL)
# fh.close()

# fh = open(f'{savePath}models','wb')
# pickle.dump(results, fh, pickle.HIGHEST_PROTOCOL)
# fh.close()