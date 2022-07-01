import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
from time import time
from resourceAllocator import MemoryResourceAllocator
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# Define the train function outside the class
# so that models can be trained in parallel w multiprocessing
def train(model):
    model.train()
    return model


class Experiment:
    def __init__(
        self,
        lmda=0.1,
        sigmaBase=5,
        learnPMT=False,
        delta_pmt=4,
        delta_1=4,
        delta_2=1,
        varySubjectParams=False,
        nSubjectsPerModel=10,
        printUpdates=False,
        adaptDelta=False,
        stepSize_adaptDelta=1,
        nTrials_adaptDelta=20,
    ) -> None:

        # Initialize variables
        self.models = []
        self.modelTypes = ["dra", "freq-s", "stakes", "equalPrecision"]
        nModels = nSubjectsPerModel * len(self.modelTypes)
        self.printUpdates = printUpdates
        self.stepSize_adaptDelta = stepSize_adaptDelta
        self.nTrials_adaptDelta = nTrials_adaptDelta
        self.lmda = lmda
        self.sigmaBase = sigmaBase
        self.delta_pmt = delta_pmt

        # set models' internal params (subject params)
        if varySubjectParams:
            lmdas = np.random.lognormal(np.log(lmda), np.log(10) / 8, nModels)
            sigmaBases = np.random.lognormal(np.log(sigmaBase), np.log(10) / 8, nModels)
        else:
            lmdas = [lmda] * nModels
            sigmaBases = [sigmaBase] * nModels

        self.subjectParams = zip(
            np.repeat(self.modelTypes, nSubjectsPerModel), lmdas, sigmaBases
        )

        # define models
        for (modelType, lmda, sigmaBase) in self.subjectParams:
            model = MemoryResourceAllocator(
                model=modelType,
                lmda=lmda,
                sigmaBase=sigmaBase,
                delta_1=delta_1,
                delta_2=delta_2,
                delta_pmt=delta_pmt,
                printUpdates=False,
                learnPMT=learnPMT,
                adaptDelta=adaptDelta,
                stepSize_adaptDelta=stepSize_adaptDelta,
                nTrials_adaptDelta=nTrials_adaptDelta,
            )
            self.models.append(model)

    # Run the experiment
    def run(self):
        # Start the timer
        start = time()

        # train all models in parallel
        pool = mp.Pool()
        self.models = pool.map(train, self.models)
        pool.close()
        pool.join()

        # Printing
        if self.printUpdates:
            timeTaken = str(dt.timedelta(seconds=time() - start))
            print(f"Finished experiment in {timeTaken}.")

    @staticmethod
    def gather_regular_choice_data_for_model(model) -> pd.DataFrame:
        states = model.choice // 3 + 1

        df = pd.DataFrame(
            {
                "Episode": range(len(states)),
                "State": states,
                "Choice accuracy": model.outcome,
                "Training-trial": [1] * model.env.episodes_train
                + [0] * model.env.episodes_pmt,
                "Task": ["Regular"] * len(states),
                "Model": [model.model] * len(states),
            }
        )
        return df.loc[df["Training-trial"] == 0].reset_index()

    @staticmethod
    def gather_bonus_choice_data_for_model(model) -> pd.DataFrame:
        states = model.choicePMT % 12 // 3 + 1
        df = pd.DataFrame(
            {
                "Episode": range(len(states)),
                "State": states,
                "Choice accuracy": model.outcomePMT,
                "Training-trial": [0] * len(states),
                "Task": ["Bonus"] * len(states),
                "Model": [model.model] * len(states),
            }
        )
        return df.dropna().reset_index()

    @staticmethod
    def preprocess_choice_data(df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.groupby(["State", "Model", "Task"])
            .agg({"Choice accuracy": "mean"})
            .reset_index()
        )
        return df

    def get_choice_data_summary(self) -> pd.DataFrame:

        list_choice_dfs = []

        for model in self.models:
            df_choices_regular = self.gather_regular_choice_data_for_model(model)
            df_choices_bonus = self.gather_bonus_choice_data_for_model(model)
            df_choices_regular = self.preprocess_choice_data(df_choices_regular)
            df_choices_bonus = self.preprocess_choice_data(df_choices_bonus)
            list_choice_dfs.extend([df_choices_regular, df_choices_bonus])

        return pd.concat(list_choice_dfs)


def plot_task_predictions(
    lmda: float = 0.1, sigmaBase: float = 5, delta_pmt: float = 4
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    experiment = Experiment(lmda=lmda, sigmaBase=sigmaBase, delta_pmt=delta_pmt)
    print("Running the experiment")
    experiment.run()
    print("Done!\nPlotting the results.")
    df = experiment.get_choice_data_summary()

    sns.relplot(
        data=df.reset_index(),
        x="State",
        y="Choice accuracy",
        hue="Model",
        col="Task",
        kind="line",
    )
    plt.xticks([1, 2, 3, 4], ["$s_1$", "$s_2$", "$s_3$", "$s_4$"])
    plt.suptitle(
        f"$\lambda$={experiment.lmda}, $\sigma_b$={experiment.sigmaBase}, $\Delta$={experiment.delta_pmt}",
        y=0.98,
    )
    plt.savefig(
        f"../figures/model_predictions_both_tasks/delta_{delta_pmt}_lmda_{int(lmda*100)}_sb_{sigmaBase}"
    )
    plt.close()


if __name__ == "__main__":
    import itertools

    lmdas = [0.01, 0.05, 0.1]
    sigmaBases = [1, 3, 5, 10]
    deltas = [2, 3, 4, 6]
    counter = 1

    for lmda, sigmaBase, delta_pmt in itertools.product(lmdas, sigmaBases, deltas):

        print(f"\nExperiment {counter} / {len(lmdas) * len(sigmaBases) * len(deltas)}")
        print(f"delta = {delta_pmt}, lmda = {lmda}, sigma_base = {sigmaBase}")

        plot_task_predictions(lmda=lmda, sigmaBase=sigmaBase, delta_pmt=delta_pmt)

        counter += 1
