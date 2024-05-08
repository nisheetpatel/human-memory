import datetime as dt
import multiprocessing as mp
from dataclasses import asdict, dataclass
from time import time
from typing import Union

import numpy as np
import pandas as pd

from src.simulation.models import Agent
from src.simulation.simulator import Simulator
from src.simulation.task import SlotMachinesTask


@dataclass
class RAModelParams:
    lmda: float = 0.05
    sigma_base: float = 5.
    lr_s: float = 0.05
    lr_v: float = 0.05


@dataclass
class RLModelParams:
    lr_v: float = 0.05
    alpha: float = 1.


@dataclass
class BIOModelParams:
    mu_0: float = 1
    kappa_0: float = 1
    alpha_0: float = 1
    beta_0: float = 1
    lambda_val: float = 0.1


ParamClass = Union[RAModelParams, RLModelParams, BIOModelParams]


class ParamGenerator:
    def __init__(self, param_class: ParamClass, N: int = 100) -> None:
        self.param_class = param_class
        self.params = param_class()
        self.N = N

    def _get_arg_vals(self):
        return [self.params.__getattribute__(key) for key in self.params.__match_args__]
        
    def _get_params_class(self, param_vals: np.ndarray):
        return self.param_class(*param_vals)

    def generate_lognormal_params(self):
        args = self._get_arg_vals()
        # param_dist = np.random.lognormal(np.log(args), 0.5, (self.N, len(args)))
        if self.param_class == BIOModelParams:
            param_dist = np.random.normal(args, 0, (self.N, len(args)))
        return [self._get_params_class(param_vals) for param_vals in param_dist]


# define method to train models in parallel
def train(simulator: Simulator) -> pd.DataFrame:
    simulator.data = simulator.train_agent(record_data=True)
    return simulator


def get_param_class(model_class: Agent) -> ParamClass:
    if model_class.__name__.endswith("RA"):
        return RAModelParams
    elif model_class.__name__.endswith("RL"):
        return RLModelParams
    elif model_class.__name__.endswith("BIO"):
        return BIOModelParams
    else:
        raise ValueError(f"No known param class for {model_class}")


class Experiment:
    def __init__(self, model_class: Agent, n_params: int = 100, n_episodes: int = 1_000) -> None:
        # generate distribution of parameters for the agent
        param_generator = ParamGenerator(get_param_class(model_class), N=n_params)
        params_list = param_generator.generate_lognormal_params()

        # define agents
        agents = [model_class(**asdict(params)) for params in params_list]

        # define simulators and model_name
        self.simulators = [Simulator(SlotMachinesTask(), agent, n_episodes) for agent in agents]
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
