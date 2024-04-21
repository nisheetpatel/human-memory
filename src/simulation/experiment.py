import datetime as dt
import multiprocessing as mp
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.simulation.models import DRA, RL, Agent, EqualRA, FreqRA, MaxEntRL, StakesRA
from src.simulation.simulator import Simulator
from src.simulation.task import SlotMachinesTask


def train(simulator: Simulator) -> pd.DataFrame:
    simulator.data = simulator.train_agent(record_data=True)
    return simulator


class Experiment:
    def __init__(self, model_class: Agent = DRA, n_runs: int = 16) -> None:
        envs = [SlotMachinesTask() for _ in range(n_runs)]
        agents = [model_class() for _ in range(n_runs)]
        self.simulators = [Simulator(env, agent) for env, agent in zip(envs, agents)]
        self.model_name = model_class.__name__

    def run(self) -> None:
        # Start the timer
        start = time()

        # train all models in parallel
        pool = mp.Pool()  # pylint: disable=consider-using-with
        self.simulators = pool.map(train, self.simulators)
        pool.close()
        pool.join()

        # print
        time_taken = str(dt.timedelta(seconds=time() - start))
        print(f"Finished training {self.model_name} in {time_taken}.")

    def extract_choice_data(self) -> pd.DataFrame:
        data = [s.data[int(len(s.data)/4) :] for s in self.simulators]
        data = pd.concat(data, ignore_index=True).reset_index()
        data["Model"] = self.model_name

        return data
    
    def extract_model_params(self):
        sigmas = [simulator.agent.sigma for simulator in self.simulators]
        return sigmas


def main():
    dfs_choice = []
    dfs_params = []
    for model_class in [DRA, FreqRA, StakesRA, EqualRA, RL, MaxEntRL]:
        exp = Experiment(model_class, n_runs=1)
        exp.run()
        dfs_choice += [exp.extract_choice_data()]
        if model_class in [DRA, FreqRA, StakesRA, EqualRA]:
            dfs_params += [exp.extract_model_params()]
    df_choice = pd.concat(dfs_choice)
    print(dfs_params)

    sns.set(font_scale=2)
    fig = sns.relplot(
        data=df_choice,
        x="price",
        y="action",
        hue="sm_id",
        style="sm_id",
        col="Model",
        kind="line",
        palette=sns.color_palette(n_colors=4),
    )
    fig.set(xticks=df_choice["price"].unique())
    # plt.show()
    plt.savefig("./model-predictions.png")
    plt.close()


if __name__ == "__main__":
    main()