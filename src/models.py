from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Optional

import numpy as np

from customtypes import Action, ActionSpace, Experience, ModelName, State
from resourceAllocators import DynamicResourceAllocator, MemoryResourceAllocator
from simulator import act_and_step
from tasks import Environment


class Agent(ABC):
    @abstractmethod
    def act(self, state: State, action_space: ActionSpace) -> Action:
        """Select an action."""
        pass

    @abstractmethod
    def update_values(self, e: Experience) -> None:
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
class ModelParams:
    sigma_base: float = 5
    gamma: float = 1
    beta: float = 10
    lmda: float = 0.1
    lr: float = 0.1
    n_trajectories: int = 10


@dataclass
class NoisyQAgent(ABC):
    model: ModelName
    q_size: tuple[int]
    resource_allocator: Optional[MemoryResourceAllocator]
    p: ModelParams = ModelParams()

    def __post_init__(self):
        self.q = np.zeros(self.q_size)
        self.sigma = self.p.sigma_base * np.ones(self.q_size)

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
            return state
        if (action is None) & (action_space is None):
            raise IndexingError("Both action and action space cannot be none.")
        if action is not None:
            return action
        if action_space is not None:
            return action_space

    def act(self, state: State, action_space: ActionSpace):
        # fetching index and defining n_actions
        n_actions = len(action_space)
        idx = self._index(state=state, action_space=action_space)

        # random draws from noisy memory distribution
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q[idx] + zeta * self.sigma[idx], self.p.beta)

        # choose action
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def update_values(self, e: Experience) -> None:
        # define indices
        idx_sa = self._index(state=e["state"], action=e["action"])
        idx_s1 = self._index(state=e["next_state"])

        # compute prediction error
        target = e["reward"] + self.p.gamma * np.max(self.q[idx_s1])
        prediction = self.q[idx_sa]
        delta = target - prediction

        # update values
        self.q[idx_sa] += self.p.lr * delta

        return

    def update_visit_counts(self, e: Experience) -> None:
        self.state_visit_counts[e["state"]] += 1
        self.action_visit_counts[e["action"]] += 1

    @abstractproperty
    def norm(self):
        """Define normalizing factor for scalar updates to noise."""

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

    def allocate_memory_resources_base(self, env: Environment):
        """Update agent's noise (sigma) parameters."""

        grads = []

        for _ in range(self.n_trajectories):
            # Initializing some variables
            grad = self.initialize_grad()  # np.zeros(self.sigma.shape)
            done = False
            state = env.reset()

            while not done:

                exp, done, zeta = act_and_step(self, env, state)

                idx_sa = self._index(state=state, action=exp["action"])
                idx_s = self._index(state=state, action_space=env.action_space)

                # advantage function
                psi = self.q[idx_sa] - np.dot(self.q[idx_s], exp["prob_actions"])

                # gradients
                self.update_grad()
                # grad[idx_s] -= (self.beta * zeta * exp["prob_actions"]) * psi
                # grad[idx_sa] += psi * self.beta * zeta[exp["action"]]

                # Update state and total reward obtained
                state = exp["next_state"]

            # collect sampled stoch. gradients for all trajectories
            grads += [grad]

        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.sigma[12:] = self.p.sigma_base

        # Compute average gradient across sampled trajs & cost
        # grad_cost = self.sigma / (self.sigma_base**2) - 1 / self.sigma
        grad_cost = self.compute_grad_cost()
        grad_mean = np.mean(grads, axis=0)

        # Updating sigmas
        self.update_noise()
        # self.sigma += self.p.lr * (grad_mean - self.p.lmda * grad_cost)

        # reset the original state
        env._episode -= self.n_trajectories
        self.sigma[12:] = 0

        return


@dataclass
class DRA(NoisyQAgent):
    model = ModelName.DRA
    resource_allocator = DynamicResourceAllocator

    @property
    def norm(self):
        return 1

    def initialize_grad(self):
        return np.zeros(self.sigma.shape)

    def update_grad(
        self, grad, psi: float, zeta: float, idx_s: int, idx_sa: int, exp: Experience
    ):
        grad[idx_s] -= (self.p.beta * zeta * exp["prob_actions"]) * psi
        grad[idx_sa] += psi * self.p.beta * zeta[exp["action"]]
        return grad

    def compute_grad_cost(self):
        return self.sigma / (self.p.sigma_base**2) - 1 / self.sigma

    def update_noise(self, delta):
        self.sigma += self.p.lr * delta


@dataclass
class FreqRA(NoisyQAgent):
    model = ModelName.FREQ
    resource_allocator = DynamicResourceAllocator

    @property
    def norm(self):
        norm_factor = np.sqrt(self.state_visit_counts)
        return 12 * norm_factor / np.sum(norm_factor[:12])

    def initialize_grad(self):
        return 0

    def update_grad(
        self, grad, psi: float, zeta: float, idx_s: int, idx_sa: int, exp: Experience
    ):
        grad -= psi * (
            self.p.beta * np.dot(exp["zeta"] / self.norm[idx_s], exp["prob_actions"])
        )
        grad += psi * (self.p.beta * zeta[exp["action_idx"]]) / self.norm[idx_sa]
        return grad

    def compute_grad_cost(self):
        return self.sigma / (self.p.sigma_base**2) - 1 / self.sigma

    def update_noise(self, delta):
        self.sigma += self.p.lr * delta


@dataclass
class StakesRA(NoisyQAgent):
    model = ModelName.STAKES
    resource_allocator = DynamicResourceAllocator

    @property
    def norm(self):
        norm_factor = np.zeros(len(self.sigma))
        norm_factor[:12] = np.tile(np.repeat([4, 1], 3), 2)
        return 12 * norm_factor / np.sum(norm_factor[:12])


@dataclass
class EqualRA(NoisyQAgent):
    model = ModelName.EQUALPRECISION
    resource_allocator = DynamicResourceAllocator

    @property
    def norm(self):
        norm_factor = np.ones(len(self.sigma))
        return 12 * norm_factor / np.sum(norm_factor[:12])
