from abc import ABC, abstractmethod

import numpy as np

from customtypes import Experience
from models import NoisyQAgent
from simulator import act_and_step
from tasks import Environment


class MemoryResourceAllocator(ABC):
    agent: NoisyQAgent
    env: Environment

    @abstractmethod
    def initialize_grad(self):
        """Initialize scalar or array-type gradient."""

    @abstractmethod
    def update_grad(self, psi: float, zeta: float, ids: list[int, int]):
        """Update gradient for the agent"""

    @abstractmethod
    def compute_grad_cost(self):
        """Compute gradient of the cost term for the agent."""

    @abstractmethod
    def update_noise(self, delta):
        """Update agent's noise vector."""

    def run(self):
        """Update agent's noise (sigma) parameters."""

        grads = []

        for _ in range(self.agent.n_trajectories):
            # Initializing some variables
            grad = self.initialize_grad()  # np.zeros(agent.sigma.shape)
            done = False
            state = self.env.reset()

            while not done:

                exp, done, zeta = act_and_step(self.agent, self.env, state)

                idx_sa = self.agent._index(state=state, action=exp["action"])
                idx_s = self.agent._index(
                    state=state, action_space=self.env.action_space
                )

                # advantage function
                psi = self.agent.q[idx_sa] - np.dot(
                    self.agent.q[idx_s], exp["prob_actions"]
                )

                # gradients
                grad = self.update_grad(grad, psi, zeta, idx_s, idx_sa, exp)
                # grad[idx_s] -= (agent.beta * zeta * exp["prob_actions"]) * psi
                # grad[idx_sa] += psi * agent.beta * zeta[exp["action"]]

                # Update state and total reward obtained
                state = exp["next_state"]

            # collect sampled stoch. gradients for all trajectories
            grads += [grad]

        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.agent.sigma[12:] = self.agent.sigma_base

        # Compute average gradient across sampled trajs & cost
        # grad_cost = agent.sigma / (agent.sigma_base**2) - 1 / agent.sigma
        grad_cost = self.compute_grad_cost()
        grad_mean = np.mean(grads, axis=0)

        # Updating sigmas
        delta = grad_mean - self.agent.p.lmda * grad_cost
        self.update_noise(delta)
        # agent.sigma += agent.p.lr *

        # reset the original state
        self.env._episode -= self.agent.n_trajectories
        self.agent.sigma[12:] = 0

        return


class DynamicResourceAllocator(MemoryResourceAllocator):
    agent: NoisyQAgent

    def initialize_grad(self):
        return np.zeros(self.agent.sigma.shape)

    def update_grad(
        self, grad, psi: float, zeta: float, idx_s: int, idx_sa: int, exp: Experience
    ):
        grad[idx_s] -= (self.agent.p.beta * zeta * exp["prob_actions"]) * psi
        grad[idx_sa] += psi * self.agent.p.beta * zeta[exp["action"]]
        return grad

    def compute_grad_cost(self):
        return self.agent.sigma / (self.agent.p.sigma_base**2) - 1 / self.agent.sigma

    def update_noise(self, delta):
        """Update agent's noise vector."""


def allocate_memory_resources(agent: NoisyQAgent, env: Environment):
    """Update agent's noise (sigma) parameters."""

    grads = []

    for _ in range(agent.p.n_trajectories):
        # Initializing some variables
        grad = np.zeros(agent.sigma.shape)
        done = False
        state = env.reset()

        while not done:

            exp, done, zeta = act_and_step(agent, env, state)

            idx_sa = agent._index(state=state, action=exp["action"])
            idx_s = agent._index(state=state, action_space=env.action_space)

            # advantage function
            psi = agent.q[idx_sa] - np.dot(agent.q[idx_s], exp["prob_actions"])

            # gradients
            # gradient_updates()
            grad[idx_s] -= (agent.p.beta * zeta * exp["prob_actions"]) * psi
            grad[idx_sa] += psi * agent.p.beta * zeta[exp["action"]]

            # Update state and total reward obtained
            state = exp["next_state"]

        # collect sampled stoch. gradients for all trajectories
        grads += [grad]

    # Setting fixed and terminal sigmas to sigma_base to avoid
    # divide by zero error; reset to 0 at the end of the loop
    agent.sigma[12:] = agent.p.sigma_base

    # Compute average gradient across sampled trajs & cost
    grad_cost = agent.sigma / (agent.p.sigma_base**2) - 1 / agent.sigma
    grad_mean = np.mean(grads, axis=0)

    # Updating sigmas
    agent.sigma += agent.p.lr * (grad_mean - agent.p.lmda * grad_cost)

    # reset the original state
    env._episode -= agent.p.n_trajectories
    agent.sigma[12:] = 0

    return


def allocate_other_models(agent: NoisyQAgent, env: Environment):
    grads = []

    for _ in range(agent.p.n_trajectories):

        # Initialising some variables
        grad = 0
        done = False
        state = env.reset()

        while not done:
            exp, done, zeta = act_and_step(agent, env, state)

            idx_sa = agent._index(state=state, action=exp["action"])
            idx_s = agent._index(state=state, action_space=env.action_space)

            # compute advantage function
            psi = agent.q[idx_sa] - np.dot(agent.q[idx_s], exp["prob_actions"])

            # gradients
            grad -= psi * (
                agent.p.beta * np.dot(zeta / agent.norm[idx_s], exp["prob_actions"])
            )
            grad += (
                psi
                * (agent.p.beta * zeta[np.array(env.action_space) == exp["action"]])
                / agent.norm[idx_sa]
            )

            # Update state for next step, add total reward
            state = exp["next_state"]

        grads += [float(grad)]

    # Setting fixed and terminal sigmas to sigma_base to avoid
    # divide by zero error; reset to 0 at the end of the loop
    agent.sigma[12:] = agent.p.sigma_base

    # Compute average gradient across sampled trajs & cost
    grad_cost = agent.sigma / (agent.p.sigma_base**2) - 1 / agent.sigma
    grad_mean = np.mean(grads, axis=0)

    # Updating sigmas
    agent.sigma += agent.p.lr * (grad_mean - agent.p.lmda * grad_cost)

    # reset the original state
    env._episode -= agent.p.n_trajectories
    agent.sigma[12:] = 0

    ##################

    # Compute average gradient across sampled trajs & cost
    grad_cost = -12 / agent.sigma_scalar + agent.sigma_scalar / (
        agent.p.sigma_base**2
    ) * np.sum(
        np.minimum(
            1 / agent.norm[:12], agent.p.sigma_base**2 / agent.sigma_scalar**2
        )
    )
    grad_mean = np.mean(grads, axis=0)

    # Updating sigmas
    agent.sigma_scalar += agent.lr * (grad_mean - agent.p.lmda * grad_cost)
    agent.sigma_scalar = np.clip(agent.sigma_scalar, 0.5, agent.p.sigma_base)

    # sigma_scalar can be noisy; so we want a moving average of it
    agent.sigma_sc_t[env.episode] = agent.sigma_scalar
    agent.sigma[:12] = np.mean(agent.sigma_sc_t[-25 : env.episode]) / agent.norm[:12]

    # reset the original state
    env._episode -= agent.p.n_trajectories
    agent.sigma[12:] = 0


# def allocate_memory_resources_base(agent: NoisyQAgent, env: Environment):
#     """Update agent's noise (sigma) parameters."""

#     grads = []

#     for _ in range(agent.n_trajectories):
#         # Initializing some variables
#         grad = initialize_grad(agent) # np.zeros(agent.sigma.shape)
#         done = False
#         state = env.reset()

#         while not done:

#             exp, done, zeta = act_and_step(agent, env, state)

#             idx_sa = agent._index(state=state, action=exp["action"])
#             idx_s = agent._index(state=state, action_space=env.action_space)

#             # advantage function
#             psi = agent.q[idx_sa] - np.dot(agent.q[idx_s], exp["prob_actions"])

#             # gradients
#             update_grad(agent)
#             # grad[idx_s] -= (agent.beta * zeta * exp["prob_actions"]) * psi
#             # grad[idx_sa] += psi * agent.beta * zeta[exp["action"]]

#             # Update state and total reward obtained
#             state = exp["next_state"]

#         # collect sampled stoch. gradients for all trajectories
#         grads += [grad]

#     # Setting fixed and terminal sigmas to sigma_base to avoid
#     # divide by zero error; reset to 0 at the end of the loop
#     agent.sigma[12:] = agent.sigma_base

#     # Compute average gradient across sampled trajs & cost
#     # grad_cost = agent.sigma / (agent.sigma_base**2) - 1 / agent.sigma
#     grad_cost = compute_grad_cost(agent)
#     grad_mean = np.mean(grads, axis=0)

#     # Updating sigmas
#     update_noise(agent)
#     # agent.sigma += agent.p.lr * (grad_mean - agent.p.lmda * grad_cost)

#     # reset the original state
#     env._episode -= agent.n_trajectories
#     agent.sigma[12:] = 0

#     return
