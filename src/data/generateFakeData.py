import itertools
import numpy as np
import pandas as pd
from experiment import Experiment

# Defining parameter set to train on
n_restarts  = 5
lmdas 		= [0.025, 0.05, 0.1, 0.15, 0.2]
sigmaBases 	= [2.5, 5, 10, 15, 20]
delta_pmts 	= np.repeat(np.array([2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7]), n_restarts)

def generateFakeData(lmdas, sigmaBases, delta_pmts, varyParams=True, adaptDelta=False):

    # initializing some variables
    subject_id = 0
    df_list = []
    expt_list = []

    # gathering choice data for param set
    for params in itertools.product(lmdas, sigmaBases, delta_pmts):

        # Defining the dictionary to pass
        paramsDict = {
            'lmda':             params[0],
            'sigmaBase':        params[1],
            'delta_pmt':        params[2]
            }
        
        # Printing to keep track
        print(f'\nExperiment {len(expt_list)+1}/{len(lmdas)*len(sigmaBases)*len(delta_pmts)}')
        print(paramsDict)

        # Running experiment
        expt = Experiment(**paramsDict, varySubjectParams=varyParams, nSubjectsPerModel=10)
        expt.run()
        choiceDataDF = expt.gatherChoiceData(subject_id=subject_id)
        df_list += [choiceDataDF]
        expt_list += [expt]

        # incrementing subject id counter
        subject_id = np.max(choiceDataDF['subject-id']) + 1

    # df = pd.concat(df_list, sort=False, ignore_index=True)

    return df_list, expt_list


if __name__ == '__main__':
    # Defining parameter set to train on
    n_restarts = 5
    paramSet = {
        'lmdas': 		[0.05, 0.1, 0.2],
        'sigmaBases': 	[5, 10],
        'delta_pmts': 	np.repeat(np.array([4]), n_restarts),
        'delta_1s': 	[4],
        'delta_2s': 	[1]
    }

    # Genrate data
    df_list, expt_list = generateFakeData(**paramSet, varyParams=True)

    # save generated data in files
    import pickle
    # fh = open('./simulatedData/choiceDataDF_merged','wb')
    # pickle.dump(df, fh, pickle.HIGHEST_PROTOCOL)
    # fh.close()

    fh = open('../data/simulatedData/choiceDataDF_varyParams_adaptDelta_list','wb')
    pickle.dump(df_list, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()

    fh = open('../data/simulatedData/experiments_varyParams_adaptDelta_list','wb')
    pickle.dump(expt_list, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()