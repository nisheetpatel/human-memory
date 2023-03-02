from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol, Type, Union

import numpy as np

from .customtypes import Action, Experience, ExperienceBuffer, ModelName, State
from .q_table import NoiseTable, NoiseTableDRA, NoiseTableScalar, QTable
from .utils import ModelParams, indexer_slots


class Agent(ABC):
    @abstractmethod
    def act(self, state: State) -> Action:
        """Select an action."""

    @abstractmethod
    def update_values(self, experience: Experience) -> None:
        """Update agent's values based on experience."""

    @abstractmethod
    def observe(self, experience: Experience) -> None:
        """Observe consequences of last action and cache them in ExperienceBuffer."""


def softargmax(x: np.ndarray, beta: float = 1) -> np.ndarray:
    """Action policy."""
    y = np.exp(beta * x - np.max(beta * x))
    return y / y.sum()


@dataclass
class NoisyQAgent(Agent):
    size: int
    model: ModelName
    q_table: QTable
    noise_table: NoiseTable
    p: ModelParams = ModelParams()
    _index: Callable = indexer_slots
    exp_buffer: ExperienceBuffer = field(default_factory=list, repr=False)

    def act(self, state: State):
        # fetch index
        idx = self._index(state=state)
        n_actions = len(idx)

        # draw from noisy memory distribution and determine action probabilities
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(
            self.q_table.values[idx] + zeta * self.noise_table.values[idx], self.p.beta
        )

        # choose action randomly given action probabilities
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def observe(self, experience: Experience) -> None:
        if experience["state"] < 16:
            self.exp_buffer += [experience]

    def update_values(self, experience: Experience) -> None:
        self.q_table.update(experience)

    def allocate_memory_resources(self) -> None:
        self.noise_table.update(self.exp_buffer, self.q_table.values)


@dataclass
class AgentCreator(Protocol):
    size: int
    p: ModelParams
    norm: Union[float, np.ndarray]


def make_agent(
    obj: AgentCreator, model_name: ModelName, noise_table: Type[NoiseTable]
) -> NoisyQAgent:
    q_table = QTable(obj.size, obj.p)
    noise_table = noise_table(obj.size, obj.p, obj.norm)
    agent = NoisyQAgent(obj.size, model_name, q_table, noise_table, obj.p)
    return agent


@dataclass
class DRA:
    size: int
    p: ModelParams = ModelParams()

    def make(self) -> NoisyQAgent:
        return make_agent(self, ModelName.DRA, NoiseTableDRA)

    @property
    def norm(self) -> float:
        return 1


@dataclass
class FreqRA:
    size: int
    p: ModelParams = ModelParams()

    def make(self) -> NoisyQAgent:
        return make_agent(self, ModelName.FREQ, NoiseTableScalar)

    @property
    def norm(self) -> np.ndarray:
        norm_factor = np.array([3, 3, 1, 1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


@dataclass
class StakesRA:
    size: int
    p: ModelParams = ModelParams()

    def make(self) -> NoisyQAgent:
        return make_agent(self, ModelName.STAKES, NoiseTableScalar)

    @property
    def norm(self) -> np.ndarray:
        norm_factor = np.array([3, 1, 3, 1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


@dataclass
class EqualRA:
    size: int
    p: ModelParams = ModelParams()

    def make(self) -> NoisyQAgent:
        return make_agent(self, ModelName.EQUALPRECISION, NoiseTableScalar)

    @property
    def norm(self):
        norm_factor = np.ones(4)
        return 4 * norm_factor / np.sum(norm_factor[:4])


if __name__ == "__main__":
    print("This file is not meant to be run as a script.")
