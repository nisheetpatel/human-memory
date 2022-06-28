import os
import fnmatch
import pickle
import numpy as np
import pandas as pd
from time import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def extract_data(exptGroup, subject_id_counter=0, varyParams=False):
    
    df_list = []

    # start counting subjects
    subject_id_counter = subject_id_counter

    # data to extract
    start_idx = 50
    end_idx = 100
    if varyParams:
        start_idx += 50
        end_idx += 50

    # loop through all models in each experiment
    for expt in exptGroup.expts[0][start_idx:end_idx]:

        for model in expt.models:

            ids = ~np.isnan(model.choicePMT)
            l = len(model.choicePMT[ids])

            # initializing state columns
            states = (model.env.states_pregen % 12) // 3

            # one-hot states
            X = np.zeros((states.size, 4))
            X[np.arange(states.size),states] = 1

            # converting state labels: 0-3 --> s1-s4
            states_dict = {0:'s1', 1:'s2', 2:'s3', 3:'s4'}
            states = np.array([states_dict[state] for state in states])

            # choice accuracy (y) column
            y = np.nan * np.ones(len(model.choicePMT))

            y[model.env.pmt_trial==1] = model.choicePMT[model.env.pmt_trial==1].copy()//12
            y[model.env.pmt_trial==-1]= 1 - (model.choicePMT[model.env.pmt_trial==-1].copy()//12)/2

            # extracting PMT trials
            states  = states[ids]
            X       = X[ids].astype(int)
            y       = y[ids].astype(int)

            # dataframe for model
            df_model = pd.DataFrame({
                    'subject-id':   [subject_id_counter] * l,
                    'model-type':   [model.model] * l,
                    'lmda':         [model.lmda] * l,
                    'sigmaBase':    [model.sigmaBase] * l,
                    'delta_pmt':    [model.env.delta_pmt] * l,
                    'state':        states,
                    'x1':           X[:,0],
                    'x2':           X[:,1],
                    'x3':           X[:,2],
                    'x4':           X[:,3],
                    'y':            y
                })
            
            # add this to the big df
            # df = pd.concat([df, df_model], ignore_index=True, sort=False)
            df_list += [df_model]

            # increase counter for next subject (here: model)
            subject_id_counter += 1            

    df = pd.concat(df_list, ignore_index=True, sort=False)

    return df


def extract_exptParamsDF(filePath='./simulatedData/experimentsvaryParams_list', save=False):
    fh = open(filePath,'rb')
    exptsList = pickle.load(fh)
    fh.close()

    expt_ids = []
    lmdas, sigmaBases, delta_pmts = [], [], []

    for expt_id, expt in enumerate(exptsList):
        expt_ids    += [expt_id]
        lmdas       += [expt.models[0].lmda]
        sigmaBases  += [expt.models[0].sigmaBase]
        delta_pmts  += [expt.models[0].env.delta_pmt]

    exptParamsDF = pd.DataFrame({
        'expt_id':  expt_ids,
        'lmda':     lmdas,
        'sigmaBase':sigmaBases,
        'delta_pmt':delta_pmts
    })

    if save:
        exptParamsDF.to_pickle('./simulatedData/exptParamsDF_varyParams')

    return exptParamsDF

def extract_posteriors_and_params(lmda=0.1, sigmaBase=10):
    # get indices of experiments to analyze
    df = pd.read_pickle('./simulatedData/exptParamsDF')
    ids = df.loc[(df['lmda']==lmda) & (df['sigmaBase']==sigmaBase), 'expt_id'].to_list()
    df_expt_params = df.iloc[ids]

    # load the corresponding posteriors
    posteriors = [None] * (len(df))
    for id in ids:
        posteriors[id] = np.load(f'./simulatedData/posteriors/mu_ca_{id}.npy')

    return posteriors, df_expt_params


if __name__ == '__main__':
    # extract file and dataframe
    fh = open('./figures/lmda10_sb7/Delta4','rb') 
    exptGroup = pickle.load(fh) 
    fh.close()
    df = extract_data(exptGroup)