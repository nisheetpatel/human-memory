# stan code
import gc
import os
import fnmatch
from numpy.core.shape_base import block
import stan
import pickle
import numpy as np
import pandas as pd
from time import time
from extractFakeDataStuff import extract_data
import matplotlib.pyplot as plt
import arviz as az


def fit_posterior_and_save(df, savePath, expt_id, model='hierarchical', n_samples=2500, n_chains=4):
    # stan code blocks
    choice_code_bernoulli = """ 
    data {
        int<lower=1> N;                 // number of training samples
        int<lower=0> K;                 // number of predictors
        matrix[N, K] x;                 // matrix of predictors
        int<lower=0, upper=1> y[N];     // observed/training choice accuracy
    }
    parameters {
        vector<lower=0, upper=1>[K] beta;
    }
    model {
        vector[N] theta;
        theta = x * beta;

        for (k in 1:K) {
            beta[k] ~ beta(4,2);
        }

        y ~ bernoulli(theta);
    }
    """
    # This one works
    choice_code_bernoulliLogit = """ 
    data {
        int<lower=1> N;                 // number of training samples
        int<lower=0> K;                 // number of predictors
        matrix[N, K] x;                 // matrix of predictors
        int<lower=0, upper=1> y[N];     // observed/training choice accuracy
    }
    parameters {
        vector[K] beta;
    }
    model {
        vector[N] theta;
        theta = x * beta;

        beta ~ normal(2,2);

        y ~ bernoulli_logit(theta);
    }
    generated quantities {
        vector[K] choiceAccuracy;
        choiceAccuracy = inv_logit(beta);
    }
    """
    choice_code_bernoulliLogit_hierarchical = """ 
    data {
        int<lower=1> N;                 // number of training samples
        int<lower=0> K;                 // number of predictors
        int<lower=1> L;                 // number of levels/subjects
        int<lower=1, upper=L> ll[N];    // subject id
        row_vector[K] x[N];             // matrix of predictors
        int<lower=0, upper=1> y[N];     // observed/training choice accuracy
    }
    parameters {
        real mu[K];
        real<lower=0.01> sigma[K];
        vector[K] beta[L];
    }
    model {
        vector[N] x_beta_ll;
        for (n in 1:N) {
            x_beta_ll[n] = x[n] * beta[ll[n]];
        }

        mu    ~ normal(1.5,5);
        sigma ~ gamma(2,5);
        
        for (l in 1:L)
            beta[l] ~ normal(mu, sigma);

        y ~ bernoulli_logit(x_beta_ll);
    }
    generated quantities {
        real mu_ca[K];
        real s_ca[K];
        for (k in 1:K) {
            mu_ca[k] = inv_logit(mu[k]);
            s_ca[k]  = inv_logit(sigma[k]);
        }
    }
    """

    # extract dataframe (relevant subjects)
    fit = []
    for i,j in [(0,10),(10,20),(20,30),(30,40)]:
        df0 = df.loc[(df['subject-id']%40>=i) & (df['subject-id']%40<j)]

        y = df0['y'].astype(int)
        X = df0.loc[:, df0.columns.isin(['x1','x2','x3','x4'])]
        s_id = np.array(df0['subject-id'])+1
        s_id -= np.min(s_id)

        # defining the choice data
        choice_data =  {'N': X.shape[0],
                        'K': X.shape[1],
                        'L': len(np.unique(s_id)),
                        'y': y.values.tolist(),
                        'x': np.array(X),
                        'll':np.array(s_id)+1}

        # fit, extract parameters, and print summary
        posterior = stan.build(choice_code_bernoulliLogit_hierarchical, 
                    data=choice_data)
        samples = posterior.sample(num_chains=n_chains, num_samples=n_samples)
        fit += [samples['mu_ca'].copy()]
        del posterior, X, y, s_id, choice_data, df0, samples
        gc.collect()
    
    # save and delete
    mu_ca = np.array(fit)
    np.save(f'{savePath}/mu_ca_{int(expt_id)}.npy', mu_ca)

    del mu_ca, fit
    gc.collect()

    return 

#     # plotting posteriors
#     ax = az.plot_trace(fit[-1], var_names=['mu_ca'], compact=True, 
#             kind='trace', combined=True, lines=None)
#     ax[0,0].set_xlabel('Choice accuracy')
#     ax[0,0].set_xlim([0.5,1])
#     ax[0,0].set_title('Posterior over mean choice accuracy in each state')
#     plt.show(block=False)
# plt.show()

# # Threshold analysis
# f,ax = plt.subplots()
# pass_percent = [0]*len(fit)
# for i in range(len(fit)):
#     b = fit[i]['mu_ca']
#     thresholds = np.linspace(0,0.2,21)
#     pass_percent[i] = []

#     for th in thresholds:

#         # conditions for frequency model
#         f1 = ((b[0] - b[1]) > th)
#         f2 = ((b[2] - b[3]) > th)
#         f  = f1 + f2

#         # conditions for stakes model
#         s1 = ((b[0] - b[2]) > th)
#         s2 = ((b[1] - b[3]) > th)
#         s  = s1 + s2

#         # combined conditions
#         c  = f * s
#         pass_percent[i] += [100*np.sum(c)/len(c)]

#     ax.plot(thresholds, pass_percent[i])

# ax.set_xlabel('Threshold value')
# ax.set_ylabel('Pass percent for test')
# ax.legend(['DRA','Frequency','Stakes','Equal Precision'])
# plt.show()


# Threshold analysis
def test(b, testFor='dra', e_factor=1, th_factor=2.5):
    
    # threshold: std. dev of posterior x th_factor x 2
    stdev = np.median(np.std(b,axis=1))
    th = 2 * stdev * th_factor
    e  = 2 * stdev * e_factor

    if testFor=='dra':
        c1 = ( b[0] - np.max(b[1:],axis=0) > th)
        c2 = ( np.min(b[:3],axis=0) - b[3] > th)
        c  = c1 + c2

    elif testFor=='freq-s':
        c1 = (abs(b[0] - b[1]) < e)
        c2 = (abs(b[2] - b[3]) < e)
        c3 = (np.min(b[:2], axis=0) - np.max(b[2:],axis=0) > th)
        c  = c1 * c2 * c3

    elif testFor=='stakes':
        c1 = (abs(b[0] - b[2]) < e)
        c2 = (abs(b[1] - b[3]) < e)
        c3 = (np.min(b[[0,2],:], axis=0) - np.max(b[[1,3],:],axis=0) > th)
        c  = c1 * c2 * c3
        
    elif testFor=='equalPrecision':
        c1 = (abs(b[0] - b[1]) < e)
        c2 = (abs(b[0] - b[2]) < e)
        c3 = (abs(b[0] - b[3]) < e)
        c4 = (abs(b[1] - b[2]) < e)
        c5 = (abs(b[1] - b[3]) < e)
        c6 = (abs(b[2] - b[3]) < e)
        c  = c1 * c2 * c3 * c4 * c5 * c6

    else:
        raise ValueError('Invalid model type!')

    pass_percent = [100*np.sum(c)/len(c)]

    return pass_percent


# Function to return the posterior probability of 
# getting a hand-defined signature that is 
# characteristic of the models
def prob_of_signature(b, th_factor=2.5):
    
    posterior_prob_matrix = []

    for i in range(len(b)):

        pass_percent = []
        modelTypes = ['dra','freq-s','stakes','equalPrecision']

        for modelType in modelTypes:
            pass_percent += test(b[i], testFor=modelType, th_factor=th_factor)

        posterior_prob_matrix += [np.array(pass_percent)/np.sum(pass_percent)]

    return np.around(posterior_prob_matrix,2)


if __name__ == "__main__":
    pass