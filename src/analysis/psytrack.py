# pylint: disable=attribute-defined-outside-init
from time import perf_counter as timer

import numpy as np
import pandas as pd
import psytrack
from matplotlib import pyplot as plt

from analysis.hierarchical import clean_data


class PsytrackModel:
    def __init__(self) -> None:
        # Define parameters for psytrack optimization
        self.weights = {"bias": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1}
        self.n_weights = sum(self.weights.values())
        self._hyperparams = self._init_hyperparams()
        self.opt_list = ["sigma"]

    def _init_hyperparams(self) -> dict:
        return {
            "sigInit": [1] * self.n_weights,
            "sigma": [4] * self.n_weights,
            "sigDay": None,
        }

    @staticmethod
    def _organize_data(data: pd.DataFrame) -> dict:
        """Organize data for psytrack."""
        if "X1" not in data.columns:
            data = clean_data(data, test_only=False)
        return {
            "inputs": {
                "x1": np.asarray(data["X1"]).reshape(-1, 1),
                "x2": np.asarray(data["X2"]).reshape(-1, 1),
                "x3": np.asarray(data["X3"]).reshape(-1, 1),
                "x4": np.asarray(data["X4"]).reshape(-1, 1),
            },
            "y": np.asarray(data["Response"]),
        }

    def fit(self, data: pd.DataFrame, jump: int = 2) -> None:
        """Fit psytrack model to data."""

        print("Fitting psytrack model...")
        start = timer()
        data_dict = self._organize_data(data)
        (
            self.hyperparams,
            self.evidence,
            self.w_mode,
            self.hess_info,
        ) = psytrack.hyperOpt(
            data_dict, self._hyperparams, self.weights, self.opt_list, jump=jump
        )

        print(f"Done in {timer() - start:.2f} seconds.\n")

    def plot_weights(self):
        """Plot psytrack inferred weights."""

        x = np.arange(self.w_mode.shape[1]) + 1
        err = self.hess_info["W_std"]
        colors = ["grey"] + plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

        for i in range(self.n_weights):
            label = f"Slope {i}" if i > 0 else "Bias"
            plt.plot(x, self.w_mode[i], label=label, color=colors[i], zorder=1)
            plt.fill_between(
                x,
                self.w_mode[i] - 1.96 * err[i],
                self.w_mode[i] + 1.96 * err[i],
                alpha=0.2,
                color=colors[i],
                zorder=0,
            )
        plt.xlabel("Trial number")
        plt.ylabel("Psytrack inferred weights")
        plt.axhline(0, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 1, c="black", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 2, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 3, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.legend()
        # plt.title(f"Psytrack estimates of slopes for subject {}")
        plt.show()
