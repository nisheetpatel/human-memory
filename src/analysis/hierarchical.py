import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    data = data.loc[:, ["id", "Response", "X1", "X2", "X3", "X4"]]

    return data


def get_choice_data_dict(data: pd.DataFrame) -> dict:
    """Get choice data dict to be fed into stan."""

    y = data["Response"].astype(int)
    X = data.loc[:, data.columns.isin(["X1", "X2", "X3", "X4"])]
    p_id = data["id"]

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

    def get_choice_data_dict(self, data: pd.DataFrame) -> dict:
        """Return choice data dict to be fed into stan."""

        return get_choice_data_dict(clean_data(data))

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


def test_model_signatures(betas: np.ndarray, e_factor=1.7, th_factor=0) -> np.ndarray:
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

    return [100 * np.sum(x) / len(x) for x in [dra, dra2, freq, stakes, ep]]


class Classifier:
    def __init__(
        self, fit: stan.fit.Fit, equality_thresh: float = 1.7, class_thresh: float = 20
    ) -> None:
        self.samples = fit["beta"]
        self.equality_thresh = equality_thresh
        self.class_thresh = class_thresh
        self.perf_metrics = pd.DataFrame(
            columns=["DRAx", "DRA+", "Freq", "Stakes", "EP", "class"]
        )
        self.class_map = {0: "DRA", 1: "DRA", 2: "Freq", 3: "Stakes", 4: "EP", 5: None}

    def classify(self):
        for i, beta in enumerate(self.samples):
            model_signatures = test_model_signatures(beta, self.equality_thresh) + [
                self.class_thresh
            ]
            model_class = self.class_map[np.argmax(model_signatures)]
            self.perf_metrics.loc[i] = model_signatures[:-1] + [model_class]
        return self.perf_metrics

    def merge_with_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.perf_metrics.merge(df, left_index=True, right_index=True)

    def get_ids(self, model: str) -> list:
        return list(
            self.perf_metrics.loc[self.perf_metrics["class"] == model].index + 1
        )


class Plotter:
    def __init__(self, fit: stan.fit.Fit, save_path: str) -> None:
        self.fit = fit
        self.n_subjects, self.n_slots, self.n_samples = self.fit["beta"].shape
        self.save_path = save_path

    def _get_df_for_posterior_plots(self) -> pd.DataFrame:
        betas = self.fit["beta"]
        slot_machine_ids = np.tile(
            np.repeat(np.arange(1, self.n_slots + 1), self.n_samples), self.n_subjects
        )
        participant_ids = np.repeat(
            np.arange(1, self.n_subjects + 1), self.n_slots * self.n_samples
        )
        return pd.DataFrame(
            {
                "id": participant_ids,
                "slot_machine_id": slot_machine_ids,
                "beta": betas.flatten(),
            }
        )

    def plot_posterior_over_betas_individual(
        self, save_name: str, perf: pd.DataFrame = None
    ) -> None:
        n_rows, n_cols = (self.n_subjects + 9) // 10, 10
        df_plot = self._get_df_for_posterior_plots()
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 8))

        for i, ax in enumerate(axs.flatten()[: self.n_subjects]):
            sns.kdeplot(
                data=df_plot[df_plot["id"] == i + 1],
                x="beta",
                hue="slot_machine_id",
                palette="tab10",
                ax=ax,
                legend=False,
            )
            if i < len(axs.flatten()) - 10:
                ax.set_xlabel("")
            if i % 10 != 0:
                ax.set_ylabel("")
            ax.set_yticklabels("")
            if perf is not None:
                ax.set_title(
                    f"{i+1}, {perf.loc[i, 'class']}, {perf.loc[i, 'performance']:.1f}"
                )

        fig.tight_layout()

        plt.savefig(self.save_path + save_name + ".png")
        plt.close()

    def plot_posteriors_at_group_level(self, save_name: str) -> None:
        df_plot = pd.DataFrame(
            {
                "slot_machine_id": np.repeat(np.arange(4) + 1, self.n_samples),
                "beta": np.reshape(self.fit["mu_beta"], -1),
                "alpha": np.reshape(self.fit["mu_alpha"], -1),
            }
        )

        _, axs = plt.subplots(2)
        for x, ax in zip(["alpha", "beta"], axs):
            sns.kdeplot(
                data=df_plot, x=x, hue="slot_machine_id", palette="tab10", ax=ax
            )
            ax.set_title(f"Group-level posterior over {x}")

        plt.savefig(self.save_path + save_name + ".png")
        plt.close()
