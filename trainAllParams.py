import pickle
import itertools
from experiment import ExperimentGroup

# params to try
lmdas = [0.1]           # [0.1, 0.03, 0.3]
sigmaBases = [5]        # [5,1,10]
delta_1s = [4]
delta_2s = [1]
delta_pmts = [0.5,2,4]
varySubjectParams = [False,True]

# expt groups
exptGroups = []

for params in itertools.product(lmdas,sigmaBases,delta_1s,delta_2s,delta_pmts,varySubjectParams):
    paramsDict = {
        'lmda':             params[0],
        'sigmaBase':        params[1],
        'delta_pmt':        params[4],
        'delta_1':          params[2],
        'delta_2':          params[3],
        'varySubjectParams':params[5]
        }
    
    print(f'Running param set {paramsDict} \n')

    exptGroup = ExperimentGroup()
    exptGroup.params = paramsDict
    exptGroup = ExperimentGroup(**paramsDict)
    exptGroup.run()
    
    exptGroups.append(exptGroup)

fh = open(f'./figures/exptGroups')
pickle.dump(exptGroups, fh, pickle.HIGHEST_PROTOCOL)
fh.close()