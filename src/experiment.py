import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
from time import time
from resourceAllocator import MemoryResourceAllocator
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# Define the train function outside the class 
# so that models can be trained in parallel w multiprocessing
def train(model):
	model.train()
	return model

class Experiment:
	def __init__(self, lmda=0.1, sigmaBase=5, learnPMT=False, delta_pmt=4, 
				delta_1=4, delta_2=1, varySubjectParams=False,
				nSubjectsPerModel=10, printUpdates=False, adaptDelta=False,
				stepSize_adaptDelta=1, nTrials_adaptDelta=20) -> None:

		# Initialize variables
		self.models = []
		self.modelTypes = ['dra', 'freq-s', 'stakes', 'equalPrecision']
		nModels = nSubjectsPerModel * len(self.modelTypes)
		self.printUpdates = printUpdates
		self.stepSize_adaptDelta = stepSize_adaptDelta
		self.nTrials_adaptDelta  = nTrials_adaptDelta

		# set models' internal params (subject params)
		if varySubjectParams:
			lmdas = np.random.lognormal(np.log(lmda), np.log(10)/8, nModels)
			sigmaBases = np.random.lognormal(np.log(sigmaBase), np.log(10)/8, nModels)
		else:
			lmdas = [lmda] * nModels
			sigmaBases = [sigmaBase] * nModels

		self.subjectParams = zip(np.repeat(self.modelTypes, nSubjectsPerModel), lmdas, sigmaBases)

		# define models
		for (modelType, lmda, sigmaBase) in self.subjectParams:
			model = MemoryResourceAllocator(\
				model        = modelType,
				lmda         = lmda,
				sigmaBase    = sigmaBase,
				delta_1      = delta_1,
				delta_2      = delta_2,
				delta_pmt    = delta_pmt,
				printUpdates = False,
				learnPMT     = learnPMT,
				adaptDelta   = adaptDelta,
				stepSize_adaptDelta = stepSize_adaptDelta,
				nTrials_adaptDelta  = nTrials_adaptDelta)
			self.models.append(model)


	# Run the experiment
	def run(self):
		# Start the timer
		start = time()

		# train all models in parallel
		pool = mp.Pool()
		self.models = pool.map(train, self.models)
		pool.close()
		pool.join()

		# Printing
		if self.printUpdates:
			timeTaken = str(dt.timedelta\
						(seconds=time()-start) )
			print(f'Finished experiment in {timeTaken}.')


	# compute dataframes of interest
	def gatherData(self):
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


	# gather choice data
	def gatherChoiceData(self, subject_id=0):
		
		df_list = []

		# start counting subjects
		subject_id = subject_id

		for model in self.models:

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

			# outcome (in/correct) (y) column
			y = np.nan * np.ones(len(model.choicePMT))

			y[model.env.pmt_trial==1] = model.choicePMT[model.env.pmt_trial==1].copy()//12
			y[model.env.pmt_trial==-1]= 1 - (model.choicePMT[model.env.pmt_trial==-1].copy()//12)/2

			# extracting PMT trials
			states  = states[ids]
			X       = X[ids].astype(int)
			y       = y[ids].astype(int)

			# dataframe for model
			df_model = pd.DataFrame({
					'subject-id':   [subject_id] * l,
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
			
			# discard data from trials when delta is still being adapted
			if model.adaptDelta:
				df_model = df_model[model.nTrials_adaptD:]
			
			# add this to the list of dataframes
			df_list += [df_model]

			# increase counter for next subject (here: model)
			subject_id += 1

		# concatenate dataframes for all models
		df = pd.concat(df_list, ignore_index=True, sort=False)

		return df


	# compute t-statistic and p-value 
	def computeStats(self, expt_id=0, varySubjectParams=False):
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
			'Model':       model,
			'State combo': stateCombos,
			'Model comp.': modelComparison,
			't-statistic': tStats,
			'p-value':     pvals,
			'Experiment':  expt_id,
			'vary subject params': bool(varySubjectParams)
		})

		# p-values for model comparison based on individual state differences
		self.df_pvals_specific = self.df_stats.groupby(['Model', 'Model comp.', 'Experiment', 'vary subject params'])\
											.agg({'p-value': 'min'})\
											.reset_index()
		
		# defining the controls: 
		# freq best in s1 vs s2 and s3 vs s4 or stakes best in s1 vs s3 or s2 vs s4
		self.df_pvals_control = self.df_pvals_specific.drop([0,1,2,3,5,6]).drop(['Model comp.'], axis=1)
		self.df_pvals_control['Test'] = 'control' 

		self.df_pvals_specific.drop([2,3,4,7], inplace=True)

		# p-values for model vs all other models
		self.df_pvals = self.df_pvals_specific.groupby(['Model', 'Experiment', 'vary subject params'])\
											.agg({'p-value': 'max'}).reset_index()
		self.df_pvals['Test'] = 'test'

		# Final df with all the info
		self.df_pvals = pd.concat([self.df_pvals,self.df_pvals_control],\
						sort=False, ignore_index=True)

		return self.df_stats, self.df_pvals



class ExperimentGroup:
	def __init__(self, nExperiments=50, savePath='./figures/', **params):

		self.nExperiments = nExperiments
		self.params = params

		# Define path and name to save results
		self.savePath = f"{savePath}" +\
			f"lmda{int(self.params['lmda']*100)}_sb{self.params['sigmaBase']}/"
		self.saveName = self.savePath + f"Delta{self.params['delta_pmt']}"
		
		# create directory if it doesn't exist
		Path(self.savePath).mkdir(parents=True, exist_ok=True)

		# Initialize two lists of experiments: w/o and w/ varying subject params
		self.expts = [[None]*self.nExperiments] * 2

		# Initialize dataframes to collect results across expts
		self.df_pvals = pd.DataFrame({
			'Experiment': 			[],
			'Model':      			[],
			'p-value':    			[],
			'vary subject params':  []
		})
		self.df_stats = pd.DataFrame({
			'Experiment': 			[],
			'Model':      			[],
			'State combo':			[],
			'Model comp.':			[],
			't-statistic':			[],
			'p-value':    			[],
			'vary subject params':  []
		})


	def run(self):
		for varySubjectParams in [False,True]:

			print(f'\n Vary subject params = {varySubjectParams}')
            
			# Run many experiments
			for i in range(self.nExperiments):
				start = time()

				expt = Experiment(varySubjectParams=varySubjectParams, **self.params)
				expt.run() 
				expt.gatherData() 
				expt.df_stats, expt.df_pvals = expt.computeStats(expt_id=i, 
									varySubjectParams=varySubjectParams)

				# collect results across experiments in dfs
				self.df_pvals = pd.concat([self.df_pvals, expt.df_pvals],
								ignore_index=True, sort=False)

				self.df_stats = pd.concat([self.df_stats, expt.df_stats],
								ignore_index=True, sort=False)

				# Printing
				timeTaken = str(dt.timedelta(seconds=time()-start) )
				print(f'{i+1}/{self.nExperiments}: Finished experiment in {timeTaken}.')

				# Add experiment to memory and clear the variable
				self.expts[varySubjectParams].append(expt)


	def save_results(self):
		fh = open(self.saveName,'wb')
		pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)
		fh.close()

		print(f'Successfully saved results.\n')

	
	def plot_results(self):
		
		# Plotting test results for defined metric for all models
		sns.set_palette('Set2')

		fig, axs = plt.subplots(1,3, figsize=(16,6))
		fig.suptitle('$\Delta_{PMT}$ = ' + str(self.params['delta_pmt']))
		
		for modelType,ax in zip(['dra', 'stakes', 'freq-s'], axs.flat):

			df = self.df_pvals[(self.df_pvals['Model']==modelType) &\
							(self.df_pvals['Test']=='test')]
			sns.histplot(df, x='p-value', hue='vary subject params', \
				log_scale=True, bins=20, ax=ax, stat='density')
			ax.set_title(modelType)
			ax.axvline(0.05,linestyle='--',color='gray')
		
		plt.savefig(f'{self.saveName}_test.svg')
		plt.close()


		# Same test on other models: control
		sns.set_palette('Paired')

		fig, axs = plt.subplots(1,2, figsize=(10,6))
		fig.suptitle('$\Delta_{PMT}$ = ' + str(self.params['delta_pmt']) + ':  Controls')
		
		for modelType,ax in zip(['stakes', 'freq-s'], axs.flat):

			df = self.df_pvals[(self.df_pvals['Model']==modelType) &\
							(self.df_pvals['Test']=='control')]
			sns.histplot(df, x='p-value', hue='vary subject params', \
				log_scale=True, bins=20, ax=ax, stat='density')
			ax.axvline(0.05,linestyle='--',color='gray')
			ax.set_title(modelType)

		plt.savefig(f'{self.saveName}_control.svg')
		plt.close()