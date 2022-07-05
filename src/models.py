from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from customtypes import Action, Experience, ExperienceBuffer, ModelName, State
from q_table import NoiseTable, NoiseTableDRA, NoiseTableScalar, QTable
from utils import ModelParams, indexer_2afc


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

    @abstractmethod
    def update_visit_counts(self, experience: Experience) -> None:
        """Update agent's visit counts based on experience."""


def softargmax(x: np.ndarray, beta: float = 1) -> np.ndarray:
    """Action policy."""
    x = np.array(x)
    b = np.max(beta * x)
    y = np.exp(beta * x - b)
    return y / y.sum()


@dataclass
class NoisyQAgent(Agent):
    q_size: int
    model: ModelName
    p: ModelParams = ModelParams()
    _index: Callable = indexer_2afc

    def __post_init__(self, noise_table: NoiseTable):
        self.q_table = QTable(size=self.q_size, p=self.p)
        self.noise_table = noise_table

        # visit counters
        self.action_visit_counts = np.ones(self.q_size, dtype=int)
        self.state_visit_counts = np.ones(self.q_size, dtype=int)

        # initializing experience buffer
        self.exp_buffer: ExperienceBuffer = []

    def act(self, state: State):
        # fetch index
        idx = self._index(state=state)
        n_actions = len(idx)

        # draw from noisy memory distribution and determine action probabilities
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q_table.values[idx] + zeta * self.noise_table.values[idx], self.p.beta)

        # choose action randomly given action probabilities
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def observe(self, experience: Experience) -> None:
        if experience["state"] < 12:
            self.exp_buffer += [experience]

    def update_values(self, experience: Experience) -> None:
        self.q_table.update(experience)

    def update_visit_counts(self, experience: Experience) -> None:
        self.state_visit_counts[experience["state"]] += 1
        self.action_visit_counts[experience["action"]] += 1

    @property
    def norm(self):
        return 1

    def allocate_memory_resources(self):
        self.noise_table.norm = self.norm
        self.noise_table.update(self.exp_buffer, self.q_table.values)


@dataclass
class DRA(NoisyQAgent):
    model: ModelName = ModelName.DRA

    def __post_init__(self):
        noise_table = NoiseTableDRA(self.q_size, self.p, self.norm)
        super().__post_init__(noise_table)

    @property
    def norm(self) -> float:
        return 1

@dataclass
class FreqRA(NoisyQAgent):
    model: ModelName = ModelName.FREQ
    
    def __post_init__(self):
        # visit counters
        self.action_visit_counts = np.ones(self.q_size, dtype=int)
        self.state_visit_counts = np.ones(self.q_size, dtype=int)

        noise_table = NoiseTableScalar(self.q_size, self.p, self.norm)
        super().__post_init__(noise_table)

    @property
    def norm(self):
        norm_factor = np.sqrt(self.state_visit_counts)
        return 12 * norm_factor / np.sum(norm_factor[:12])


@dataclass
class StakesRA(NoisyQAgent):
    model: ModelName = ModelName.STAKES
    
    def __post_init__(self):
        noise_table = NoiseTableScalar(self.q_size, self.p, self.norm)
        super().__post_init__(noise_table)

    @property
    def norm(self):
        norm_factor = np.tile(np.repeat([4, 1], 3), 2)
        return 12 * norm_factor / np.sum(norm_factor[:12])


@dataclass
class EqualRA(NoisyQAgent):
    model: ModelName = ModelName.EQUALPRECISION

    def __post_init__(self):
        noise_table = NoiseTableScalar(self.q_size, self.p, self.norm)
        super().__post_init__(noise_table)

    @property
    def norm(self):
        norm_factor = np.ones(12)
        return 12 * norm_factor / np.sum(norm_factor[:12])


if __name__ == "__main__":
    print("This file is not meant to be run as a script.")
