import pickle
import numpy as np
import pandas as pd
import itertools
from dra_rangel import DynamicResourceAllocator
from time import time
import datetime

# Define the train function outside the class
def train(model):
    model.train()
    return model

class Experiment:
    def __init__(self, lmda=0.1, sigmaBase=5, learnPMT=False, delta_pmt=2, 
                delta_1=4, delta_2=1, varySubjectParams=False,
                nSubjectsPerModel=10) -> None:

        # Define modelTypes to train
        modelTypes = ['dra', 'freq-s', 'stakes', 'equalPrecision']
        self.modelTypes = modelTypes

        # Initialize the models
        self.models = []
        nModels = nSubjectsPerModel * len(modelTypes)

        if varySubjectParams:
            lmdas = np.random.lognormal(np.log(lmda), np.log(10)/4, nModels)
            sigmaBases = np.random.lognormal(np.log(sigmaBase), np.log(10)/4, nModels)
        else:
            lmdas = [lmda] * nModels
            sigmaBases = [sigmaBase] * nModels

        self.subjectParams = zip(np.repeat(modelTypes, nSubjectsPerModel), lmdas, sigmaBases)

        for (modelType, lmda, sigmaBase) in self.subjectParams:
            model = DynamicResourceAllocator(\
                model        = modelType,\
                lmda         = lmda,\
                sigmaBase    = sigmaBase,\
                delta_1      = delta_1,\
                delta_2      = delta_2,\
                delta_pmt    = delta_pmt,\
                printUpdates = False,\
                learnPMT     = learnPMT)
            self.models.append(model)


    # Run the experiment
    def run(self):
        import multiprocessing as mp
        start = time()
        pool = mp.Pool()
        self.models = pool.map(train, self.models)

        # Printing
        timeTaken = str(datetime.timedelta\
                    (seconds=time()-start) )
        print(f'Finished experiment in {timeTaken}.')


    # compute dataframes of interest
    def gatherData(self, save=False):
        # initializing list of models and empty df
        new_models = []
        self.df = pd.DataFrame({'state':[], 
                                'p(choose better option)':[],
                                'pmt-type':[],
                                'model':[]
        })

        for model in self.models:
            # Compute probability that the model picks better option on PMT trials
            model.pChooseBetter = np.zeros(24)

            for a in range(12):
                # +Delta
                idx = (np.where((model.env.pmt_trial>0) & (model.choicePMT==a)))[0] 
                model.pChooseBetter[a] = 1-len(idx)/(model.env.n_pmt/2)
            
                # -Delta
                idx = (np.where((model.env.pmt_trial<0) & (model.choicePMT==a)))[0]  
                model.pChooseBetter[a+12] = len(idx)/(model.env.n_pmt/2)

            # Create dataframe with p(choose better option) for each state and action
            model.df_p = pd.DataFrame({
                'state':                    np.tile(np.repeat(['s1','s2','s3','s4'],3),2),
                'action':                   np.tile(model.env.actions[:12]%3,2),
                'p(choose better option)':  model.pChooseBetter,
                'pmt-type':                 ['+']*12 + ['-']*12
            })

            # average over actions in each state (so variability does not get carried over)
            model.df_p = model.df_p.groupby(['state', 'pmt-type'])\
                          .agg({'p(choose better option)': 'mean'})\
                          .reset_index()
            
            # add model column and append it to new_models
            model.df_p['model'] = model.model
            new_models.append(model)

            # concatenate all models' dfs
            self.df = pd.concat( [self.df, model.df_p], \
                        ignore_index=True, sort=False)

        # update models in experiment
        self.models = new_models


    # compute t-statistic and p-value 
    def computeStats(self, save=False):
        from scipy import stats

        stateCombos, modelComparison, model, tStats, pvals = [], [], [], [], []

        for modelType in self.modelTypes:
            for (si,sj) in [('s1','s2'),('s3','s4'), ('s1','s3'),('s2','s4')]:
                p_si = self.df.loc [ (self.df['state']==si) & (self.df['model']==modelType), 'p(choose better option)'] 
                p_sj = self.df.loc [ (self.df['state']==sj) & (self.df['model']==modelType), 'p(choose better option)']
                result = stats.ttest_rel(p_si, p_sj)

                if (si,sj) in [('s1','s2'),('s3','s4')]:
                    modelComp = 'freq'
                elif (si,sj) in [('s1','s3'),('s2','s4')]:
                    modelComp = 'stakes'

                stateCombos += [f'{si} vs {sj}']
                modelComparison += [modelComp]
                model  += [modelType]
                tStats += [result[0]]
                pvals  += [result[1]]
        
        self.df_stats = pd.DataFrame({
            'Model':             model,
            'State combination': stateCombos,
            'Model comparison':  modelComparison,
            't-statistic':       tStats,
            'p-value':           pvals
        })

        # p-values for model comparison based on individual state differences
        self.df_pvals_specific = self.df_stats.groupby(['Model', 'Model comparison'])\
                                            .agg({'p-value': 'min'})\
                                            .reset_index()
        self.df_pvals_specific.drop([2,3,4,7], inplace=True)

        # p-values for model vs all other models
        self.df_pvals = self.df_pvals_specific.groupby('Model').agg({'p-value': 'max'}).reset_index()