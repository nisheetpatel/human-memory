import numpy as np

from customtypes import Experience
from utils import ModelParams, indexer_2afc


def get_indices(experience: Experience) -> tuple:
    """Compute all indices for possible operations."""
    idx_s = indexer_2afc(state=experience["state"])
    idx_s1 = indexer_2afc(state=experience["next_state"])
    idx_sa = indexer_2afc(state=experience["state"], action=experience["action"])
    return idx_s, idx_s1, idx_sa


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


# @dataclass
# class NoiseTable:
#     size: int
#     p: ModelParams

#     def update_visit_counts(self, experience: Experience) -> None:
#         self.state_visit_counts[experience["state"]] += 1
#         self.action_visit_counts[experience["action"]] += 1

#     def _compute_advantage(self, experience: Experience) -> float:
#         idx_s, _, idx_sa = get_indices(experience)
#         return self.q[idx_sa] - np.dot(self.q[idx_s], experience["prob_actions"])

#     def _initialize_grad(self):
#         """Initialize scalar gradient by default."""
#         return 0

#     def _update_grad(self, grad, advantage: float, experience: Experience):
#         """Update scalar gradient for the agent"""
#         idx_s, _, idx_sa = get_indices(experience)
#         grad -= advantage * (
#             self.p.beta
#             * np.dot(experience["zeta"] / self.norm[idx_s], experience["prob_actions"])
#         )
#         grad += (
#             advantage
#             * (self.p.beta * experience["zeta"][experience["action_idx"]])
#             / self.norm[idx_sa]
#         )
#         return grad

#     def _compute_grad_cost(self):
#         """Compute scalar gradient of the cost term for the agent."""
#         grad_cost = -12 / self.sigma_scalar + self.sigma_scalar / (
#             self.p.sigma_base**2
#         ) * np.sum(
#             np.minimum(
#                 1 / self.norm[:12], self.p.sigma_base**2 / self.sigma_scalar**2
#             )
#         )
#         return grad_cost

#     def _update_noise(self, delta):
#         """Update agent's noise vector."""
#         self.sigma_scalar += self.p.lr * delta
#         self.sigma_history.append(self.sigma_scalar)

#         # sigma_scalar can be noisy; so we update with its moving average
#         self.sigma[:12] = np.mean(self.sigma_history[-25:]) / self.norm[:12]

#     def allocate_memory_resources(self):
#         """Update agent's noise (sigma) parameters."""
#         grads = []

#         for experience in self.exp_buffer[-self.p.n_trajectories :]:
#             psi = self._compute_advantage(experience)
#             grads += [self._update_grad(self._initialize_grad(), psi, experience)]

#         # Setting fixed and terminal sigmas to sigma_base to avoid
#         # divide by zero error; reset to 0 at the end of the loop
#         self.sigma[12:] = self.p.sigma_base

#         # Compute average gradient across sampled trajs & cost
#         grad_cost = self._compute_grad_cost()
#         grad_mean = np.mean(grads, axis=0)
#         delta = grad_mean - self.p.lmda * grad_cost

#         # Updating sigmas
#         self._update_noise(delta)

#         # reset the original state
#         self.sigma[12:] = 0
