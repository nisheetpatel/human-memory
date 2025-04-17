import itertools
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from src.simulation.models import DRA
from src.simulation.simulator import Simulator
from src.simulation.task import SlotMachinesTask


def compute_accuracy(df_choice: pd.DataFrame) -> float:
    df_choice["Choice Accuracy"] = 0

    df_choice.loc[((df_choice["price"] < 0) & (df_choice["action"] == 0)), "Choice Accuracy"] = 1
    df_choice.loc[((df_choice["price"] > 0) & (df_choice["action"] == 1)), "Choice Accuracy"] = 1

    return df_choice["Choice Accuracy"].mean() * 100


# define method to train models in parallel
def train(simulator: Simulator) -> pd.DataFrame:
    simulator.data = simulator.train_agent(record_data=True)
    return simulator


class Experiment:
    def __init__(self, n_episodes: int = 1_000) -> None:
        # generate distribution of parameters for the agent
        lmdas = np.logspace(-2, 0, 100)
        lrs = np.linspace(0.01, 0.5, 50)

        # define agents
        self.agents = [DRA(lmda=lmda, lr_s=lr_s) for lmda, lr_s in itertools.product(lmdas, lrs)]

        # define simulators and model_name
        self.simulators = [Simulator(SlotMachinesTask(), agent, n_episodes) for agent in self.agents]
        self.model_name = "DRA"

    def run(self) -> None:
        # train all models in parallel
        pool = mp.Pool()  # pylint: disable=consider-using-with
        self.simulators = pool.map(train, self.simulators)
        pool.close()
        pool.join()

    def extract_performance_and_params(self) -> pd.DataFrame:
        data = [s.data[int(len(s.data)/4) :] for s in self.simulators]

        agents = [s.agent for s in self.simulators]
        lmdas = [a.lmda for a in agents]
        learning_rates = [a.lr_s for a in agents]
        accuracies = [compute_accuracy(d) for d in data]

        df = pd.DataFrame({"Choice Accuracy": accuracies, "lr_s": learning_rates, "lmda": lmdas})
        df['log-lmda'] = np.log10(df["lmda"])

        return df


exp = Experiment()
exp.run()
df = exp.extract_performance_and_params()


# Plot for accuracy vs lmda
r, p = sp.stats.pearsonr(df["Choice Accuracy"], df["log-lmda"])

sns.lmplot(data=df, y="Choice Accuracy", x="log-lmda", scatter_kws={"color": "green"},
           line_kws={"color": "green"})
plt.xlabel("Memory cost ($\lambda$, log-scaled)")
plt.ylabel("Choice accuracy")

# Transform the axes for the text placement
plt.gca().text(0.85, 0.85, f"r={r:.2f}\np={p:.2e}", transform=plt.gca().transAxes,
               horizontalalignment='center', verticalalignment='top')

plt.savefig('./test.png')
plt.close()


# Learning rate
r, p = sp.stats.pearsonr(df["Choice Accuracy"], df["lr_s"])

sns.lmplot(data=df, y="Choice Accuracy", x="lr_s", scatter_kws={"color": "royalblue"},
           line_kws={"color": "royalblue"})
plt.xlabel("Learning rate for noise ($\sigma$")
plt.ylabel("Choice accuracy")

# Transform the axes for the text placement
plt.gca().text(0.85, 0.85, f"r={r:.2f}\np={p:.2e}", transform=plt.gca().transAxes,
               horizontalalignment='center', verticalalignment='top')

plt.savefig('./test.png')
plt.close()
