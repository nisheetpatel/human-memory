import sys

sys.path.append("scripts")
from resourceAllocator import MemoryResourceAllocator
import itertools
from time import time
import multiprocessing as mp
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def low_cost_pmt():
    # define range of parameters to test
    lmdas = [0.005, 0.01, 0.05, 0.1]
    sigma_bases = [1, 2.5]

    n_runs = 10
    model_types = ["dra", "freq-s", "stakes", "equalPrecision"]

    for model_type in model_types:
        models = []

        # DEFINE MODELS
        for lmda, sigma_base, _ in itertools.product(
            lmdas, sigma_bases, list(range(n_runs))
        ):
            model = MemoryResourceAllocator(
                model=model_type,
                lmda=lmda,
                sigmaBase=sigma_base,
                delta_pmt=2,
                printUpdates=False,
                learnPMT=False,
                adaptDelta=True,
            )
            models += [model]

        print(f"Starting training for {model_type}.")

        # TRAIN ALL MODELS
        def train(model):
            model.train()
            return model

        # Start the timer
        start = time()

        # train all models in parallel
        pool = mp.Pool()
        models = pool.map(train, models)
        pool.close()
        pool.join()

        # Printing
        timeTaken = str(dt.timedelta(seconds=time() - start))
        print(f"Finished training in {timeTaken}.\n")

        # GATHER CHOICE DATA
        # initializing list of models and empty df
        dfs, new_models = [], []

        for model in models:
            # Compute probability that the model picks better option on PMT trials
            model.pChooseBetter = np.zeros(24)

            for a in range(12):
                # +Delta
                idx = (np.where((model.env.pmt_trial > 0) & (model.choicePMT == a)))[0]
                model.pChooseBetter[a] = 1 - len(idx) / (model.env.n_pmt / 2)

                # -Delta
                idx = (np.where((model.env.pmt_trial < 0) & (model.choicePMT == a)))[0]
                model.pChooseBetter[a + 12] = len(idx) / (model.env.n_pmt / 2)

            # Create dataframe with p(choose better option) for each state and action
            model.df_p = pd.DataFrame(
                {
                    "state": np.tile(np.repeat(["s1", "s2", "s3", "s4"], 3), 2),
                    "action": np.tile(model.env.actions[:12] % 3, 2),
                    "p(choose better option)": model.pChooseBetter,
                    "pmt-type": ["+"] * 12 + ["-"] * 12,
                }
            )

            # average over actions in each state (so variability does not get carried over)
            model.df_p = (
                model.df_p.groupby(["state", "pmt-type"])
                .agg({"p(choose better option)": "mean"})
                .reset_index()
            )

            # add model column and append it to new_models
            model.df_p["model"] = model.model
            model.df_p["lmda"] = model.lmda
            model.df_p["sigma_base"] = model.sigmaBase
            new_models.append(model)

            # concatenate all models' dfs
            dfs += [model.df_p]

        # update models in experiment and concatenate all dfs
        models = new_models
        df = pd.concat(dfs, ignore_index=True, sort=False)

        # PLOTTING
        plt.close()
        sns.lineplot(
            x="state",
            y="p(choose better option)",
            hue="lmda",
            style="sigma_base",
            data=df,
            palette="colorblind",
        )
        plt.title(f"{model_type}")
        plt.savefig(
            f"../pilot data/low_cost_model_predictions/{model_type}_predictions_low_cost.svg"
        )
        plt.savefig(
            f"../pilot data/low_cost_model_predictions/{model_type}_predictions_low_cost.png"
        )
        plt.close()


def low_cost_regular():
    # define range of parameters to test
    lmdas = [0.01, 0.1]
    sigma_bases = [1, 2.5]

    n_runs = 10
    model_types = ["dra", "freq-s", "stakes", "equalPrecision"]

    for model_type in model_types:
        models = []

        # DEFINE MODELS
        for lmda, sigma_base, _ in itertools.product(
            lmdas, sigma_bases, list(range(n_runs))
        ):
            model = MemoryResourceAllocator(
                model=model_type,
                lmda=lmda,
                sigmaBase=sigma_base,
                delta_pmt=2,
                printUpdates=False,
                learnPMT=False,
                adaptDelta=True,
            )
            models += [model]

        print(f"Starting training for {model_type}.")

        # TRAIN ALL MODELS
        def train(model):
            model.train()
            return model

        # Start the timer
        start = time()

        # train all models in parallel
        pool = mp.Pool()
        models = pool.map(train, models)
        pool.close()
        pool.join()

        # Printing
        timeTaken = str(dt.timedelta(seconds=time() - start))
        print(f"Finished training in {timeTaken}.\n")

        # GATHER CHOICE DATA
        # initializing list of models and empty df
        dfs, new_models = [], []

        for model in models:
            # Compute probability that the model picks better option on regular trials
            model.pChooseBetter = np.zeros(12)

            for state in range(12):
                model.env.state = state
                correct = 0

                for i in range(1000):
                    action, _, _, actions = model.act(state)
                    if action == min(actions):
                        correct += 1

                model.pChooseBetter[state] = correct / 1000

                # Create dataframe with p(choose better option) for each state and action
                model.df_p = pd.DataFrame(
                    {
                        "state": np.tile(np.repeat(["s1", "s2", "s3", "s4"], 3), 1),
                        "action": np.tile(model.env.actions[:12] % 3, 2),
                        "p(choose better option)": model.pChooseBetter,
                    }
                )

            # average over actions in each state (so variability does not get carried over)
            model.df_p = (
                model.df_p.groupby(["state"])
                .agg({"p(choose better option)": "mean"})
                .reset_index()
            )

            # add model column and append it to new_models
            model.df_p["model"] = model.model
            model.df_p["lmda"] = model.lmda
            model.df_p["sigma_base"] = model.sigmaBase
            new_models.append(model)

            # concatenate all models' dfs
            dfs += [model.df_p]

        # update models in experiment and concatenate all dfs
        models = new_models
        df = pd.concat(dfs, ignore_index=True, sort=False)

        # PLOTTING
        plt.close()
        sns.lineplot(
            x="state",
            y="p(choose better option)",
            hue="lmda",
            style="sigma_base",
            data=df,
            palette="colorblind",
        )
        plt.title(f"{model_type}")
        plt.savefig(
            f"../pilot data/low_cost_model_predictions_reg/{model_type}_predictions_low_cost.svg"
        )
        plt.savefig(
            f"../pilot data/low_cost_model_predictions_reg/{model_type}_predictions_low_cost.png"
        )
        plt.close()


if __name__ == "__main__":
    low_cost_regular()
