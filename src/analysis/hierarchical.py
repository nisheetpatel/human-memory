import pickle

import numpy as np
import pandas as pd
import stan


def clean_data(data: pd.DataFrame, test_only=True) -> pd.DataFrame:
    """Clean data for Stan."""

    data = data.copy()
    data.dropna(inplace=True)
    data["Slot Machine ID"] = (data["Slot Machine ID"] + 1).astype(int)
    data["Response"] = data["Response"].astype(int)

    for i in range(1, 5):
        data[f"X{i}"] = data["Difficulty"] * (data["Slot Machine ID"] == i)

    if test_only:
        data = data.loc[data["block_type"] == "test"]

    data = data.loc[:, ["participant_id", "Response", "X1", "X2", "X3", "X4"]]

    return data


def get_choice_data_dict(data: pd.DataFrame, id_map: dict) -> dict:
    """Get choice data dict to be fed into stan."""

    y = data["Response"].astype(int)
    X = data.loc[:, data.columns.isin(["X1", "X2", "X3", "X4"])]
    p_id = data["participant_id"].map(id_map)

    return {
        "N": X.shape[0],  # number of training samples
        "K": X.shape[1],  # number of predictors
        "L": len(p_id.unique()),  # number of levels/subjects
        "y": y.values.tolist(),  # response variable
        "X": np.array(X),  # matrix of predictors
        "ll": np.array(p_id.values),  # subject id
        "ss": np.array(X != 0, dtype=int),  # slot machine id indicator
    }


############################################################
# STAN MODELS
############################################################

LOGIT_DIFF_BIAS = """
    data {
        int<lower=1> N; // number of training samples
        int<lower=0> K; // number of predictors (4 SMs)
        int<lower=1> L; // number of subjects
        
        int<lower=1, upper=L> ll[N]; // subject id {1,...,L}
        row_vector<lower=0, upper=1>[K] ss[N]; // slot machine id indicator

        row_vector[K] X[N];         // predictors
        int<lower=0, upper=1> y[N]; // response
    }

    parameters {
        vector[K] beta[L];  // individual slope
        vector[K] mu_beta;  // Hierarchical mean for slope
        vector<lower=0>[K] sigma_beta; // h std for slope
        
        vector[K] alpha[L]; // individual intercept
        vector[K] mu_alpha;   // Hierarchical mean for intercept
        vector<lower=0>[K] sigma_alpha; // h std for intercept

        vector[L] alpha_0; // common intercept
        real mu_alpha_0;   // hierarchical mean for common intercept
        real<lower=0> sigma_alpha_0; // h std for common intercept
    }

    model {
        mu_beta ~ normal(0.25, 2);
        sigma_beta ~ cauchy(0, 1);
            
        mu_alpha ~ normal(0, 0.5);
        sigma_alpha ~ cauchy(0, 0.5);
        
        mu_alpha_0 ~ normal(0, 0.5);
        sigma_alpha_0 ~ cauchy(0, 0.5);

        for (l in 1:L) {
            beta[l] ~ normal(mu_beta, sigma_beta);
            alpha[l] ~ normal(mu_alpha, sigma_alpha);
            alpha_0[l] ~ normal(mu_alpha_0, sigma_alpha_0);
        }
        
        {
        vector[N] x_beta_ll;

        for (n in 1:N)
            x_beta_ll[n] = X[n] * beta[ll[n]] + ss[n] * alpha[ll[n]] + alpha_0[ll[n]];
        
        y ~ bernoulli_logit(x_beta_ll);
        }
    }
"""

LOGIT_COMMON_BIAS = """
    data {
        int<lower=1> N; // number of training samples
        int<lower=0> K; // number of predictors (4 SMs)
        int<lower=1> L; // number of subjects
        
        int<lower=1, upper=L> ll[N]; // subject id {1,...,L}
        row_vector<lower=0, upper=1>[K] ss[N]; // slot machine id indicator

        row_vector[K] X[N];         // predictors
        int<lower=0, upper=1> y[N]; // response
    }

    parameters {
        vector[K] beta[L];  // individual slope
        vector[K] mu_beta;  // Hierarchical mean for slope
        vector<lower=0>[K] sigma_beta; // h std for slope

        vector[L] alpha_0; // common intercept
        real mu_alpha_0;   // hierarchical mean for common intercept
        real<lower=0> sigma_alpha_0; // h std for common intercept
    }

    model {
        mu_beta ~ normal(0.25, 2);
        sigma_beta ~ cauchy(0, 1);
        
        mu_alpha_0 ~ normal(0, 0.25);
        sigma_alpha_0 ~ cauchy(0, 0.5);

        for (l in 1:L) {
            beta[l] ~ normal(mu_beta, sigma_beta);
            alpha_0[l] ~ normal(mu_alpha_0, sigma_alpha_0);
        }
        
        {
        vector[N] x_beta_ll;

        for (n in 1:N)
            x_beta_ll[n] = X[n] * beta[ll[n]] + alpha_0[ll[n]];
        
        y ~ bernoulli_logit(x_beta_ll);
        }
    }
"""


class HierarchicalModel:
    """Hierarchical model for choice data."""

    def __init__(self, model: str = LOGIT_DIFF_BIAS):
        self.model = model
        self.id_map = None

    def get_choice_data_dict(self, data: pd.DataFrame) -> dict:
        """Return choice data dict to be fed into stan."""

        data = clean_data(data)
        self.id_map = {j: i + 1 for i, j in enumerate(data["participant_id"].unique())}
        return get_choice_data_dict(data, self.id_map)

    def fit_posterior(self, data: dict, n_chains=4, n_samples=10_000) -> stan.fit.Fit:
        """Build stan model and sample from posterior."""

        posterior = stan.build(self.model, data=data)
        return posterior.sample(num_chains=n_chains, num_samples=n_samples)

    def save(self, fit: stan.fit.Fit, filename: str) -> None:
        """Save fit object."""

        with open(filename, "wb") as file:
            pickle.dump({"model": self.model, "fit": fit}, file, protocol=-1)

    def load(self, filename: str) -> stan.fit.Fit:
        """Load fit object."""

        with open(filename, "rb") as file:
            return pickle.load(file)["fit"]
