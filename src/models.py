from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from customtypes import Action, ActionSpace, Experience, Reward, State


class Agent(ABC):
    @abstractmethod
    def act(self, state: State) -> Action:
        """Select an action."""
        pass

    @abstractmethod
    def update_values(self) -> None:
        """Update values based on observation."""
        pass

    # @abstractmethod
    # def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
    #     """Observe consequences of the last action."""
    #     pass


@dataclass
class IndexingError(Exception):
    error: str


def softargmax(x: np.ndarray, beta: float = 1) -> np.array:
    x = np.array(x)
    b = np.max(beta * x)
    y = np.exp(beta * x - b)
    return y / y.sum()


@dataclass
class NoisyQAgent(Agent):
    name: str
    q_size: tuple[int]
    sigma_base: float = 5
    gamma: float = 1
    beta: float = 10
    lmda: float = 0.1
    lr: float = 0.1
    n_trajectories: int = 10

    def __post_init__(self):
        self.q = np.zeros(self.q_size)
        self.sigma = self.sigma_base * np.ones(self.q_size)

        # visit counters
        self.action_visit_counts = np.zeros(self.q_size, dtype=int)
        self.state_visit_counts = np.zeros(self.q_size, dtype=int)

    def _index(
        self,
        state: State = None,
        action: Action = None,
        action_space: ActionSpace = None,
    ) -> int:
        """Indexes q-table entries."""
        if state == -1:
            return tuple(state)
        if (action is None) & (action_space is None):
            raise IndexingError("Both action and action space cannot be none.")
        if action is not None:
            return tuple(action)
        if action_space is not None:
            return action_space

    def act(self, state: State, action_space: ActionSpace):
        # fetching index and defining n_actions
        n_actions = len(action_space)
        idx = self._index(action_space=action_space)

        # random draws from noisy memory distribution
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q[idx] + zeta * self.sigma[idx], self.beta)

        # choose action
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def update_values(self, e: Experience) -> None:
        # define indices
        idx_sa = self._index(state=e["state"], action=e["action"])
        idx_s1 = self._index(state=e["next_state"])

        # compute prediction error
        target = e["reward"] + self.gamma * np.max(self.q[idx_s1])
        prediction = self.q[idx_sa]
        delta = target - prediction

        # update values
        self.q[idx_sa] += self.lr * delta

        return

    def update_visit_counts(self, e: Experience) -> None:
        self.state_visit_counts[e["state"]] += 1
        self.action_visit_counts[e["action"]] += 1
