import pickle

import pandas as pd
import stan

from src.data.stan_preparation import prepare_for_stan

from .stan_models import LOGIT_DIFF_BIAS


class HierarchicalModel:
    """Hierarchical model for choice data."""

    def __init__(self, model: str = LOGIT_DIFF_BIAS):
        self.model = model


    def get_choice_data_dict(self, data: pd.DataFrame) -> dict:
        """Return choice data dict to be fed into stan."""

        return prepare_for_stan(data=data, test_only=True)


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



