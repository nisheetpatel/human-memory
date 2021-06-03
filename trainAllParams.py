import itertools
from experiment import ExperimentGroup

# Defining parameter set to train on
paramSet = {
	'lmdas': 		[0.1, 0.05, 0.25],
	'sigmaBases': 	[5,3,7],
	'delta_pmts': 	[1,2,3,4],
	'delta_1s': 	[4],
	'delta_2s': 	[1]
}

def trainAllParams(lmdas, sigmaBases, delta_pmts, delta_1s, delta_2s,
		nExperimentsPerParamSet=50):

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

		# Running the set of n=50 Experiments
		exptGroup = ExperimentGroup(nExperiments=nExperimentsPerParamSet, **paramsDict)
		exptGroup.params = paramsDict
		exptGroup.run()
		exptGroup.save_results()
		exptGroup.plot_results()

		# Clearing unused objects from memory
		del exptGroup


if __name__ == '__main__':
	# Defining parameter set to train on
	paramSet = {
		'lmdas': 		[0.1, 0.05, 0.25],
		'sigmaBases': 	[5,3,7],
		'delta_pmts': 	[1,2,3,4],
		'delta_1s': 	[4],
		'delta_2s': 	[1]
	}
	trainAllParams(**paramSet)