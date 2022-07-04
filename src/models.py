from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from customtypes import (Action, Done, Experience, ExperienceBuffer, Info,
                         ModelName, Reward, State)
from q_table import QTable
from utils import ModelParams, indexer_2afc


class Agent(ABC):
    @abstractmethod
    def act(self, state: State) -> Action:
        """Select an action."""

    @abstractmethod
    def update_values(self, experience: Experience) -> None:
        """Update agent's values based on experience."""

    @abstractmethod
    def observe(self, obs: State, reward: Reward, done: Done, info: Info) -> None:
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
class NoisyQAgent(ABC):
    q_size: int
    model: ModelName
    p: ModelParams = ModelParams()
    _index: Callable = indexer_2afc

    def __post_init__(self):
        self.q_table = QTable(size=self.q_size, p=self.p)
        self.sigma = self.p.sigma_base * np.ones(self.q_size)
        self.sigma_scalar = 1
        self.sigma_history = []

        # visit counters
        self.action_visit_counts = np.zeros(self.q_size, dtype=int)
        self.state_visit_counts = np.zeros(self.q_size, dtype=int)

        # initializing experience buffer
        self.exp_buffer: ExperienceBuffer = []

    def act(self, state: State):
        # fetch index
        idx = self._index(state=state)
        n_actions = len(idx)

        # draw from noisy memory distribution and determine action probabilities
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q_table.values[idx] + zeta * self.sigma[idx], self.p.beta)

        # choose action randomly given action probabilities
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def observe(self, experience: Experience) -> None:
        self.exp_buffer += [experience]

    def _compute_indices(self, experience: Experience):
        """Compute all indices for possible operations."""
        idx_s = self._index(state=experience["state"])
        idx_s1 = self._index(state=experience["next_state"])
        idx_sa = self._index(state=experience["state"], action=experience["action"])
        return idx_s, idx_s1, idx_sa

    def update_values(self, experience: Experience) -> None:
        self.q_table.update(experience)

    def update_visit_counts(self, experience: Experience) -> None:
        self.state_visit_counts[experience["state"]] += 1
        self.action_visit_counts[experience["action"]] += 1

    @property
    @abstractmethod
    def norm(self):
        """Define normalizing factor for scalar updates to noise."""

    def _compute_advantage(self, experience: Experience) -> float:
        idx_s, _, idx_sa = self._compute_indices(experience)
        return self.q_table.values[idx_sa] - np.dot(self.q_table.values[idx_s], experience["prob_actions"])

    def _initialize_grad(self):
        """Initialize scalar gradient by default."""
        return 0

    def _update_grad(self, grad, advantage: float, experience: Experience):
        """Update scalar gradient for the agent"""
        idx_s, _, idx_sa = self._compute_indices(experience)
        grad -= advantage * (
            self.p.beta
            * np.dot(experience["zeta"] / self.norm[idx_s], experience["prob_actions"])
        )
        grad += (
            advantage
            * (self.p.beta * experience["zeta"][experience["action_idx"]])
            / self.norm[idx_sa]
        )
        return grad

    def _compute_grad_cost(self):
        """Compute scalar gradient of the cost term for the agent."""
        grad_cost = -12 / self.sigma_scalar + self.sigma_scalar / (
            self.p.sigma_base**2
        ) * np.sum(
            np.minimum(
                1 / self.norm[:12], self.p.sigma_base**2 / self.sigma_scalar**2
            )
        )
        return grad_cost

    def _update_noise(self, delta):
        """Update agent's noise vector."""
        self.sigma_scalar += self.p.lr * delta
        self.sigma_history.append(self.sigma_scalar)

        # sigma_scalar can be noisy; so we update with its moving average
        self.sigma[:12] = np.mean(self.sigma_history[-25:]) / self.norm[:12]

    def allocate_memory_resources(self):
        """Update agent's noise (sigma) parameters."""
        grads = []

        for experience in self.exp_buffer[-self.p.n_trajectories :]:
            psi = self._compute_advantage(experience)
            grads += [self._update_grad(self._initialize_grad(), psi, experience)]

        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.sigma[12:] = self.p.sigma_base

        # Compute average gradient across sampled trajs & cost
        grad_cost = self._compute_grad_cost()
        grad_mean = np.mean(grads, axis=0)
        delta = grad_mean - self.p.lmda * grad_cost

        # Updating sigmas
        self._update_noise(delta)

        # reset the original state
        self.sigma[12:] = 0


@dataclass
class DRA(NoisyQAgent):
    model: ModelName = ModelName.DRA

    @property
    def norm(self):
        return 1

    def _initialize_grad(self):
        return np.zeros(self.sigma.shape)

    def _update_grad(self, grad, advantage: float, experience: Experience):
        idx_s, _, idx_sa = self._compute_indices(experience)
        grad[idx_s] -= (
            advantage * self.p.beta * experience["zeta"] * experience["prob_actions"]
        )
        grad[idx_sa] += (
            advantage * self.p.beta * experience["zeta"][experience["action"]]
        )
        return grad

    def _compute_grad_cost(self):
        return self.sigma / (self.p.sigma_base**2) - 1 / self.sigma

    def _update_noise(self, delta):
        self.sigma += self.p.lr * delta


@dataclass
class FreqRA(NoisyQAgent):
    model: ModelName = ModelName.FREQ

    @property
    def norm(self):
        norm_factor = np.sqrt(self.state_visit_counts)
        return 12 * norm_factor / np.sum(norm_factor[:12])


@dataclass
class StakesRA(NoisyQAgent):
    model: ModelName = ModelName.STAKES

    @property
    def norm(self):
        norm_factor = np.zeros(len(self.sigma))
        norm_factor[:12] = np.tile(np.repeat([4, 1], 3), 2)
        return 12 * norm_factor / np.sum(norm_factor[:12])


@dataclass
class EqualRA(NoisyQAgent):
    model: ModelName = ModelName.EQUALPRECISION

    @property
    def norm(self):
        norm_factor = np.ones(len(self.sigma))
        return 12 * norm_factor / np.sum(norm_factor[:12])


if __name__ == "__main__":
    print("This file is not meant to be run as a script.")
