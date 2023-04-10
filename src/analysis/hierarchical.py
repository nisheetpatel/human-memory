import pickle

import numpy as np
import pandas as pd
import stan

from analysis import stan_models


def clean_data(data: pd.DataFrame, test_only=True) -> pd.DataFrame:
    """Clean data for Stan."""

    data = data.copy()
    data.dropna(inplace=True)
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


class HierarchicalModel:
    """Hierarchical model for choice data."""

    def __init__(self, model: str = stan_models.LOGIT_DIFF_BIAS):
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


def test_model_signatures(betas: np.ndarray, e_factor=1, th_factor=0) -> np.ndarray:
    """
    Test percentage of samples from the posterior over betas that
    pass the test for all models in customtype.ModelName.
    """

    stdev = np.median(np.std(betas, axis=1))
    th = stdev * th_factor
    e = stdev * e_factor

    dra = (
        (betas[0] - betas[1] > th)
        * (betas[0] - betas[2] > th)
        * (betas[0] - betas[3] > th)
        * (betas[1] - betas[3] > th)
        * (betas[2] - betas[3] > th)
    )
    freq = (
        (abs(betas[0] - betas[1]) < e)
        * (abs(betas[2] - betas[3]) < e)
        * ((betas[0] - betas[2]) > th)
        * ((betas[0] - betas[3]) > th)
        * ((betas[1] - betas[2]) > th)
        * ((betas[1] - betas[3]) > th)
    )
    stakes = (
        (abs(betas[0] - betas[2]) < e)
        * (abs(betas[1] - betas[3]) < e)
        * ((betas[0] - betas[1]) > th)
        * ((betas[0] - betas[3]) > th)
        * ((betas[2] - betas[1]) > th)
        * ((betas[2] - betas[3]) > th)
    )
    ep = (
        (abs(betas[0] - betas[1]) < e)
        * (abs(betas[0] - betas[2]) < e)
        * (abs(betas[0] - betas[3]) < e)
        * (abs(betas[1] - betas[2]) < e)
        * (abs(betas[1] - betas[3]) < e)
        * (abs(betas[2] - betas[3]) < e)
    )
    dra2 = (
        (betas[0] - betas[3] > th)
        * (betas[0] - betas[1] > -e)
        * (betas[0] - betas[2] > -e)
        * (betas[1] - betas[3] > -e)
        * (betas[2] - betas[3] > -e)
        * np.logical_not(freq + stakes + ep)
    )

    return [lambda x: 100 * np.sum(x) / len(x) for x in [dra, dra2, freq, stakes, ep]]
