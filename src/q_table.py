from abc import ABC, abstractmethod

import numpy as np

from customtypes import Experience, ExperienceBuffer
from utils import ModelParams, indexer_2afc


def get_indices(experience: Experience) -> tuple:
    """Compute all indices for possible operations."""
    idx_s = indexer_2afc(state=experience["state"])
    idx_s1 = indexer_2afc(state=experience["next_state"])
    idx_sa = indexer_2afc(state=experience["state"], action=experience["action"])
    return idx_s, idx_s1, idx_sa

def compute_advantage(experience: Experience, q_values: np.ndarray) -> float:
    """currently returning reward NOT advantage"""
    idx_s, _, idx_sa = get_indices(experience)
    return q_values[idx_sa] - np.dot(q_values[idx_s], experience["prob_actions"])


class QTable:
    def __init__(self, size: int, p: ModelParams):
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
        self.values[idx_sa] += self.p.lr * delta


class NoiseTable(ABC):
    @abstractmethod
    def update(self, experience_buffer: ExperienceBuffer, q_values: np.ndarray):
        """Update noise table"""


class NoiseTableDRA(NoiseTable):
    def __init__(self, size: int, p: ModelParams, norm):
        self.p = p
        self.values = self.p.sigma_base * np.ones(size)
        self.norm = norm

    def _initialize_grad(self):
        """Initialize scalar gradient by default."""
        return np.zeros(self.values.shape)

    def _update_grad(self, grad, advantage: float, experience: Experience):
        """Update scalar gradient for the agent"""
        idx_s, _, idx_sa = get_indices(experience)
        grad[idx_s] -= (
            advantage * self.p.beta * experience["zeta"] * experience["prob_actions"]
        )
        grad[idx_sa] += (
            advantage * self.p.beta * experience["zeta"][experience["action"]]
        )
        return grad

    def _compute_grad_cost(self):
        """Compute scalar gradient of the cost term for the agent."""
        return self.values / (self.p.sigma_base**2) - 1 / self.values

    def _update_noise(self, delta):
        """Update agent's noise vector."""
        self.values += self.p.lr * delta

    def update(self, experience_buffer: ExperienceBuffer, q_values: np.ndarray):
        """Update agent's noise (sigma) parameters."""
        grads = []

        for experience in experience_buffer[-self.p.n_trajectories :]:
            psi = compute_advantage(experience, q_values)
            grads += [self._update_grad(self._initialize_grad(), psi, experience)]

        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.values[12:] = self.p.sigma_base

        # Compute average gradient across sampled trajs & cost
        grad_cost = self._compute_grad_cost()
        grad_mean = np.mean(grads, axis=0)
        delta = grad_mean - self.p.lmda * grad_cost

        # Updating sigmas
        self._update_noise(delta)

        # Clip sigma to be less than sigma_base
        self.values = np.clip(self.values, 0.01, self.p.sigma_base)

        # reset the original state
        self.values[12:] = 0


class NoiseTableScalar(NoiseTable):
    def __init__(self, size: int, p: ModelParams, norm):
        self.p = p
        self.values = self.p.sigma_base * np.ones(size)
        self.sigma_scalar = 1
        self.sigma_history = []
        self.norm = norm

    def _compute_advantage(self, experience: Experience) -> float:
        """currently returning reward NOT advantage"""
        return experience["reward"]

    def _initialize_grad(self):
        """Initialize scalar gradient by default."""
        return 0

    def _update_grad(self, grad, advantage: float, experience: Experience):
        """Update scalar gradient for the agent"""
        idx_s, _, idx_sa = get_indices(experience)
        grad -= advantage * (
            self.p.beta
            * np.dot(experience["zeta"] / self.norm[idx_s], experience["prob_actions"])
        )
        grad += (
            advantage
            * (self.p.beta * experience["zeta"][experience["action"]])
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
        self.values[:12] = np.mean(self.sigma_history[-25:]) / self.norm[:12]

    def update(self, experience_buffer: ExperienceBuffer, q_values: np.ndarray):
        """Update agent's noise (sigma) parameters."""
        grads = []

        for experience in experience_buffer[-self.p.n_trajectories :]:
            psi = compute_advantage(experience, q_values)
            grads += [self._update_grad(self._initialize_grad(), psi, experience)]

        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.values[12:] = self.p.sigma_base

        # Compute average gradient across sampled trajs & cost
        grad_cost = self._compute_grad_cost()
        grad_mean = np.mean(grads, axis=0)
        delta = grad_mean - self.p.lmda * grad_cost

        # Updating sigmas
        self._update_noise(delta)

        # Clip sigma to be less than sigma_base
        self.values = np.clip(self.values, 0.01, self.p.sigma_base)

        # reset the original state
        self.values[12:] = 0
