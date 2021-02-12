import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# import multiprocessing as mp
# from train import train_model, save_results

""" Figure better, cleaner way to do this"""
# Redefining parameters
# n_restarts = 10
keyStates  = [5, 6, 16, 17] 
keyStates0 = [5, 16]
correctAct = [7, 9, 18, 20]
keyActions = [7, 8, 9, 10, 18, 19, 20, 21]


# # Defining model types to train
modelTypes = ['dra', 'equalPrecision', 'freq 1',\
            'freq 0.97', 'freq 0.95', 'freq 0.9']

# # Running in parallel
# pool = mp.Pool()
# results = pool.map(train_model, \
#     np.repeat(modelTypes, n_restarts))

# savePath = save_results(results)

# define path to load files from
savePath = f'./figures/2_v1/'
df = pickle.load(f'{savePath}df_before')


# Plotting Objective, expected reward, cost
for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df, x="Episode", y=toPlot,\
        hue="Model", style="Model", dashes=False)
    plt.savefig(f'{savePath}Perf_{toPlot}')
    plt.close()

for modelType in modelTypes: 
    fig, ax = plt.subplots() 
    sns.lineplot(x='Episode', y='Sigma', hue='Action',\
        data=df[df.Model==modelType], legend='full',\
        palette='Paired')
    ax.set_ylim([0,50])
    
    if modelType.split()[0] == 'freq':
        saveName = 'freq' + str(int((float\
            (modelType.split()[1])*100)))
    else:
        saveName = modelType.split()[0]
    
    plt.savefig(f'{savePath}Sigma_{saveName}_evolution')
    plt.close()

# Plotting all things at steady state
dfPlot = df[df.Episode >= (np.max(df.Episode)-100)]
# df2 = df[df.Episode>=(episodes+extraEps-100)]

for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x="Model", y=toPlot)
    plt.savefig(f'{savePath}Perf_{toPlot}_steadyState_beforeSwitch')
    plt.close()

for toPlot in ['P_visit', 'P_mistake', 'Choice accuracy']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x="State", y=toPlot, hue='Model')
    ax.set_title('Before switch')
    plt.savefig(f'{savePath}Obs_{toPlot}_steadyState_beforeSwitch')
    plt.close()

# Creating a column of actions: 0 and 1 for split plotting
col_action = np.array(dfPlot['Action'])
for a in keyActions:
    if a in correctAct:
        col_action[col_action == a] = 0
    else:
        col_action[col_action == a] = 1
dfPlot['action'] = col_action

for modelType in modelTypes:
    
    dfModel = dfPlot[dfPlot.Model == modelType]

    if modelType.split()[0] == 'freq':
        saveName = 'freq' + str(int((float\
            (modelType.split()[1])*100)))
        palette = 'Set3'
    else:
        saveName = modelType.split()[0]
        if modelType == 'dra':
            palette = 'Set1'
        else:
            palette = 'Set2'

    fig, ax = plt.subplots()
    ax = sns.violinplot(data=dfModel, x='State', y='Sigma',\
        hue='action', split=True, palette=palette)
    ax.set_title(f'{modelType} - Before switch')
    plt.savefig(f'{savePath}Sigma_{saveName}_steadyState_beforeSwitch')
    plt.close()



# Plotting after
# Plotting all things at steady state
dfPlot = df[df.Episode >= (1200)]
# df2 = df[df.Episode>=(episodes+extraEps-100)]

for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x="Model", y=toPlot)
    plt.savefig(f'{savePath}Perf_{toPlot}_steadyState_afterSwitch')
    plt.close()

for toPlot in ['P_visit', 'P_mistake', 'Choice accuracy']:
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=dfPlot, x="State", y=toPlot, hue='Model')
    ax.set_title('After switch')
    plt.savefig(f'{savePath}Obs_{toPlot}_steadyState_afterSwitch')
    plt.close()

# Creating a column of actions: 0 and 1 for split plotting
col_action = np.array(dfPlot['Action'])
for a in keyActions:
    if a in correctAct:
        col_action[col_action == a] = 0
    else:
        col_action[col_action == a] = 1
dfPlot['action'] = col_action

for modelType in modelTypes:
    
    dfModel = dfPlot[dfPlot.Model == modelType]

    if modelType.split()[0] == 'freq':
        saveName = 'freq' + str(int((float\
            (modelType.split()[1])*100)))
        palette = 'Set3'
    else:
        saveName = modelType.split()[0]
        if modelType == 'dra':
            palette = 'Set1'
        else:
            palette = 'Set2'

    fig, ax = plt.subplots()
    ax = sns.violinplot(data=dfModel, x='State', y='Sigma',\
        hue='action', split=True, palette=palette)
    ax.set_title(f'{modelType} - After switch')
    plt.savefig(f'{savePath}Sigma_{saveName}_steadyState_afterSwitch')
    plt.close()

# Best SIGMA
fig, ax = plt.subplots()
ax = sns.boxplot(data=dfPlot, x='Model', y='Sigma',\
    hue='Action', palette='Paired')
plt.savefig(f'{savePath}SIGMA_afterSwitch')
plt.close()

# Plotting Objective, expected reward, cost
for toPlot in ['Objective', 'Expected reward', 'Cost']:
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df, x="Episode", y=toPlot,\
        hue="Model", style="Model", dashes=False)
    plt.savefig(f'{savePath}Perf_{toPlot}_full')
    plt.close()

for modelType in modelTypes: 
    fig, ax = plt.subplots() 
    sns.lineplot(x='Episode', y='Sigma', hue='Action',\
        data=df[df.Model==modelType], legend='full',\
        palette='Paired')
    ax.set_ylim([0,50])
    
    if modelType.split()[0] == 'freq':
        saveName = 'freq' + str(int((float\
            (modelType.split()[1])*100)))
    else:
        saveName = modelType.split()[0]
    
    plt.savefig(f'{savePath}Sigma_{saveName}_evolution_full')
    plt.close()

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


# # Re-pasting inputs for current, unclean code
# # Because I might need to re-load df, models
# import numpy as np
# import pandas as pd
# from task import BottleneckTask
# from dra import DynamicResourceAllocator
# import multiprocessing as mp
# from time import time
# import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# # Redefining modelTypes
# modelTypes = ['dra', 'equalPrecision', 'freq 1',\
#              'freq 0.9', 'freq 0.75', 'freq 0.5']