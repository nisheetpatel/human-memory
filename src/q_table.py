from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from customtypes import Experience, ExperienceBuffer
from utils import ModelParams, indexer_slots


def get_indices(experience: Experience) -> tuple:
    """Compute all indices for possible operations."""
    idx_s = indexer_slots(state=experience["state"])
    idx_s1 = indexer_slots(state=experience["next_state"])
    idx_sa = indexer_slots(state=experience["state"], action=experience["action"])
    return idx_s, idx_s1, idx_sa


def compute_advantage(experience: Experience, q_values: np.ndarray) -> float:
    """currently returning reward NOT advantage"""
    idx_s, _, idx_sa = get_indices(experience)
    return q_values[idx_sa] - np.dot(q_values[idx_s], experience["prob_actions"])


class QTable:
    def __init__(self, size: int, p: ModelParams) -> None:
        self.values = np.zeros(size)
        self.p = p

    def update(self, experience: Experience) -> None:
        # compute relevant indices
        _, idx_s1, idx_sa = get_indices(experience)

        # compute prediction error
        target = experience["reward"] + self.p.gamma * np.max(self.values[idx_s1])
        prediction = self.values[idx_sa]
        delta = target - prediction

        # update values
        if target != 0:
            self.values[idx_sa] += self.p.lr * delta
        self.values[:4] = 0


class NoiseTable(ABC):
    values: np.ndarray
    p: ModelParams

    @abstractmethod
    def _initialize_grad(self) -> Union[float, np.ndarray]:
        """Initialize gradient."""

    @abstractmethod
    def _compute_grad(self, grad, advantage: float, exp: Experience):
        """Update gradient for the agent"""

    @abstractmethod
    def _compute_grad_cost(self) -> Union[float, np.ndarray]:
        """Compute gradient of the cost term for the agent."""

    @abstractmethod
    def _update_noise(self, delta) -> None:
        """Update noise vector given delta."""

    def update(self, exp_buffer: ExperienceBuffer, q_values: np.ndarray) -> None:
        """Update noise table by sampling trajectories from experience buffer."""
        grads = []

        for experience in exp_buffer[-self.p.n_trajectories :]:
            psi = compute_advantage(experience, q_values)
            grads += [self._compute_grad(self._initialize_grad(), psi, experience)]

        # Setting fixed terms to sigma_base to avoid cost and divide by zero error
        self.values[4:] = self.p.sigma_base

        # Compute average gradient across sampled trajs & cost, then update
        delta = np.mean(grads, axis=0) - self.p.lmda * self._compute_grad_cost()
        self._update_noise(delta)

        # Clip sigma to be less than sigma_base, then reset fixed ones to 0
        self.values = np.clip(self.values, 0.01, self.p.sigma_base)
        self.values[4:] = 0


class NoiseTableDRA(NoiseTable):
    def __init__(self, size: int, p: ModelParams, norm):
        self.p = p
        self.values = np.ones(size)
        self.norm = norm

    def _initialize_grad(self):
        return np.zeros(self.values.shape)

    def _compute_grad(self, grad, advantage: float, exp: Experience) -> np.ndarray:
        idx_s, _, idx_sa = get_indices(exp)
        grad[idx_s] -= advantage * self.p.beta * exp["zeta"] * exp["prob_actions"]
        grad[idx_sa] += advantage * self.p.beta * exp["zeta"][exp["action"]]
        return grad

    def _compute_grad_cost(self) -> np.ndarray:
        return self.values / (self.p.sigma_base**2) - 1 / self.values

    def _update_noise(self, delta) -> None:
        self.values += self.p.lr * delta


class NoiseTableScalar(NoiseTable):
    def __init__(self, size: int, p: ModelParams, norm):
        self.p = p
        self.values = np.ones(size)
        self.sigma_scalar = 1
        self.sigma_history = []
        self.norm = norm

    def _initialize_grad(self):
        return 0

    def _compute_grad(self, grad, advantage: float, exp: Experience) -> float:
        idx_s, _, _ = get_indices(exp)
        grad -= advantage * (
            self.p.beta * np.dot(exp["zeta"] / self.norm[idx_s[0]], exp["prob_actions"])
        )
        grad += (
            advantage * (self.p.beta * exp["zeta"][exp["action"]]) / self.norm[idx_s[0]]
        )
        return grad

    def _compute_grad_cost(self) -> float:
        grad_cost = -4 / self.sigma_scalar + self.sigma_scalar / (
            self.p.sigma_base**2
        ) * np.sum(
            np.minimum(
                1 / self.norm[:4], self.p.sigma_base**2 / self.sigma_scalar**2
            )
        )
        return grad_cost

    def _update_noise(self, delta) -> None:
        self.sigma_scalar += self.p.lr * delta
        self.sigma_history.append(self.sigma_scalar)

        # sigma_scalar can be noisy; so we update with its moving average
        self.values[:4] = np.mean(self.sigma_history[-25:]) / self.norm[:4]
