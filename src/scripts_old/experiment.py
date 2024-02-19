import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass
from time import time
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simulation.models import DRA, EqualRA, FreqRA, StakesRA
from simulation.simulator import Simulator
from simulation.task import SlotMachinesTask


def train(simulator: Simulator):
    simulator.train_agent()
    return simulator


@dataclass
class Experiment:
    model_class: DRA | FreqRA | StakesRA | EqualRA = DRA
    n_runs: int = 16

    def __post_init__(self):
        envs = [SlotMachinesTask() for _ in range(self.n_runs)]
        size = envs[0].n_states
        agents = [self.model_class(size).make() for _ in range(self.n_runs)]
        simulators = [Simulator(agent, env) for agent, env in zip(agents, envs)]
        self.simulators = simulators

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
        model_name = self.simulators[0].agent.model.value
        print(f"Finished training {self.n_runs} x {model_name} in {time_taken}.")

    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df["Slot-Machine"] = df["choice_set"].apply(lambda x: x[0] + 1)
        df["Difficulty"] = df["choice_set"].apply(lambda x: x[1] - 4)
        df["Difficulty"].replace({0: -2, 1: -1, 2: 1, 3: 2}, inplace=True)
        df["Response"] = 1 - df["action"]
        df["Correct"] = 0
        df.loc[(df["Difficulty"] < 0) & (df["Response"] == 0), "Correct"] = 1
        df.loc[(df["Difficulty"] > 0) & (df["Response"] == 1), "Correct"] = 1
        return df[["Slot-Machine", "Difficulty", "Response", "Correct"]]

    def _extract_data_single_run(self, simulator: Simulator) -> pd.DataFrame:
        data = simulator.agent.exp_buffer
        assert len(data) > 0, "Cannot extract data before running the experiment."
        data = data[-10 * 192 :]
        return self._preprocess_data(pd.DataFrame(data))

    @staticmethod
    def _extract_values_single_run(simulator: Simulator) -> pd.DataFrame:
        q = simulator.agent.q_table.values
        noise = simulator.agent.noise_table.values
        return pd.DataFrame({"option": list(range(len(q))), "q": q, "noise": noise})

    def _extract(self, func: Callable) -> pd.DataFrame:
        df_list = []
        for run, simulator in enumerate(self.simulators):
            df = func(simulator)
            df["run"] = run
            df_list.append(df)

        df = pd.concat(df_list)
        df["Model"] = self.simulators[0].agent.model.value
        return df.reset_index()

    def extract_choice_data(self) -> pd.DataFrame:
        return self._extract(self._extract_data_single_run)

    def extract_values(self) -> pd.DataFrame:
        return self._extract(self._extract_values_single_run)


def main():
    dfs_choice = []
    dfs_values = []
    for model_class in [DRA, FreqRA, StakesRA, EqualRA]:
        exp = Experiment(model_class)
        exp.run()
        dfs_choice += [exp.extract_choice_data()]
        dfs_values += [exp.extract_values()]
    df_choice = pd.concat(dfs_choice)
    df_values = pd.concat(dfs_values).reset_index()
    df_values["Slot machine"] = df_values["option"] + 1
    df_values = df_values.loc[df_values["Slot machine"] <= 4]

    sns.set(font_scale=2)
    fig = sns.relplot(
        data=df_choice,
        x="Difficulty",
        y="Response",
        hue="Slot-Machine",
        style="Slot-Machine",
        col="Model",
        kind="line",
        palette=sns.color_palette(n_colors=4),
    )
    fig.set(xticks=df_choice["Difficulty"].unique())
    # plt.show()
    plt.savefig("./figures/slot-machines/model-predictions.svg")
    plt.close()

    sns.set(font_scale=2)
    fig_2 = sns.lineplot(
        data=df_values,
        x="Slot machine",
        y="noise",
        hue="Model",
        palette=sns.color_palette(n_colors=4),
    )
    fig_2.set(xticks=df_values["Slot machine"].unique())
    # plt.show()
    plt.savefig("./figures/slot-machines/true-noise.svg")
    plt.close()


if __name__ == "__main__":
    main()
