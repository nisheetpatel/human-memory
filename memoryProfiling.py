# memory profiling
import gc
from experiment import ExperimentGroup, Experiment
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
sns.set_palette('colorblind')

# Define params
defaultParams = [0.1, 5, 4, 1, 3, False]

#@profile
def trainAllParams(params=defaultParams):

    # Defining the dictionary to pass
    paramsDict = {
        'lmda':             params[0],
        'sigmaBase':        params[1],
        'delta_1':          params[2],
        'delta_2':          params[3],
        'delta_pmt':        params[4],
        'varySubjectParams':params[5]
        }
    
    # Printing to keep track
    print(f'\nRunning param set {paramsDict} \n')

    # Running the set of n=50 Experiments
    exptGroup = ExperimentGroup(nExperiments=5, **paramsDict)
    exptGroup.params = paramsDict
    exptGroup.run()

    # Clearing unused objects from memory
    del exptGroup
    gc.collect()


if __name__ == '__main__':
    # trainAllParams()
    from time import time
    import datetime
    import gc

    # Define params
    defaultParams = [0.1, 5, 4, 1, 3, False]
    params = {
        'lmda':             defaultParams[0],
        'sigmaBase':        defaultParams[1],
        'delta_1':          defaultParams[2],
        'delta_2':          defaultParams[3],
        'delta_pmt':        defaultParams[4],
        'varySubjectParams':defaultParams[5]
        }
    nExperiments = 5

    # Run many experiments
    for i in range(nExperiments):
        start = time()

        expt = Experiment(**params)   # add **params here as dict
        expt.run() 

        # Printing
        timeTaken = str(datetime.timedelta\
                    (seconds=time()-start) )
        print(f'{i+1}/{nExperiments}: Finished experiment in {timeTaken}.')

        del expt
        gc.collect()