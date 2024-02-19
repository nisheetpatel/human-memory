from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from scipy.stats import entropy, norm


class Agent(Protocol):
    def action_prob(self, sm_id: int, price: float) -> tuple:
        ...

    def act(self, sm_id: int, price: float) -> int:
        ...

    def update(self, sm_id: int, price: float, reward: float, action: int) -> None:
        ...


class DRA:
    def __init__(self, lr_v: float = 0.05, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_0: float = 2.5, sigma_base: float = 5) -> None:
        # define parameters
        self.lr_v = lr_v
        self.lr_s = lr_s
        self.lmda = lmda
        self.sigma_base = sigma_base
        
        # define initial value and noise
        self.v = np.array([0.,0.,0.,0.])
        self.sigma = np.array([1.,1.,1.,1.]) * sigma_0
        self.sigma_history = [sigma_0]

    def action_prob(self, sm_id: int, price: float):
        p_no = norm.cdf((price - self.v[sm_id]) / self.sigma[sm_id])
        return (1-p_no, p_no)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, action: int):
        grad_cost = self.sigma / self.sigma_base ** 2 - 1 / self.sigma
        grad_reward = 0

        if action == 0:
            self.v[sm_id] += self.lr_v * (reward - self.v[sm_id])

            x = (price - self.v[sm_id]) / self.sigma[sm_id]
            grad_reward = norm.pdf(x) / (1 - norm.cdf(x)+ 1e-4) * x / self.sigma[sm_id]
            grad_reward *= reward

        self.sigma[sm_id] += self.lr_s * grad_reward
        self.sigma -= self.lr_s * self.lmda * grad_cost

        self.sigma = np.clip(self.sigma, 0.01, self.sigma_base)


class OtherRA(ABC):
    def __init__(self, lr_v: float = 0.05, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_0: float = 5, sigma_base: float = 5) -> None:

        self.v = np.array([0,0,0,0])
        self.sigma_scalar = sigma_0
        self.sigma = self.sigma_scalar * np.array([1,1,1,1]) / self.norm

        self.lr_v = lr_v
        self.lr_s = lr_s
        self.lmda = lmda
        self.sigma_base = sigma_base
        self.sigma_history = []
        self.n_back = 25

    @property
    @abstractmethod
    def norm(self):
        pass

    def action_prob(self, sm_id: int, price: float):
        p_no = norm.cdf((price - self.v[sm_id]) / self.sigma[sm_id])
        return (1-p_no, p_no)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, action: bool):
        grad_cost = np.sum(self.sigma / self.sigma_base ** 2 - 1 / self.sigma)
        grad_reward = 0

        if action == 0:
            # update mean values
            self.v[sm_id] += self.lr_v * (reward - self.v[sm_id])

            # define argument to pdf and cdf for concise notation
            x = (price - self.v[sm_id]) / self.sigma[sm_id]

            # define gradient of reward term of the objective function
            grad_reward = norm.pdf(x) / (1 - norm.cdf(x) + 1e-4) * x / self.sigma[sm_id]
            grad_reward *= reward / self.norm[sm_id]

            # self.sigma[sm_id] += self.lr * grad_reward

        # update sigma_scalar and sigma
        self.sigma_scalar += self.lr_s * (grad_reward - self.lmda * grad_cost)
        self.sigma_history.append(self.sigma_scalar)
        self.sigma = np.mean(self.sigma_history[-self.n_back:]) / self.norm

        self.sigma_scalar = np.clip(self.sigma_scalar, 0.01, self.sigma_base)
        self.sigma = np.clip(self.sigma, 0.01, self.sigma_base)


class EqualRA(OtherRA):
    @property
    def norm(self):
        norm_factor = np.ones(4)
        return 4 * norm_factor / np.sum(norm_factor[:4])


class FreqRA(OtherRA):
    @property
    def norm(self):
        norm_factor = np.array([np.sqrt(3),np.sqrt(3),1,1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


class StakesRA(OtherRA):
    @property
    def norm(self):
        norm_factor = np.array([np.sqrt(3),1,np.sqrt(3),1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


def softargmax(x: np.ndarray, beta: float = 1) -> np.ndarray:
    y = np.exp(beta * x - np.max(beta * x))
    return y / y.sum()


class RL:
    def __init__(self, lr_v: float = 0.05, beta: float = 1) -> None:
        # define parameters
        self.lr_v = lr_v
        self.beta = beta

        # define initial values
        self.v = np.array([0., 0., 0., 0.])

    def action_prob(self, sm_id: int, price: float) -> np.ndarray:
        return softargmax(np.array([self.v[sm_id], price]), self.beta)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, action: int) -> None:
        if action == 0:
            self.v[sm_id] += self.lr_v * (reward - self.v[sm_id])


class MaxEntRL:
    def __init__(self, lr_v: float = 0.05, alpha: float = 1) -> None:
        # define parameters
        self.lr_v = lr_v
        self.alpha = alpha

        # define initial values
        self.q = np.zeros((4,4))
        self.v = np.zeros((4,4))
        self.p_id = {-2: 0, -1: 1, 1: 2, 2: 3}

    def action_prob(self, sm_id: int, price: float) -> np.ndarray:
        return softargmax(np.array([self.q[sm_id, self.p_id[price]],0]), 1 / self.alpha)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, action: int) -> None:
        self.q[sm_id, self.p_id[price]] += self.lr_v * (reward - self.v[sm_id, self.p_id[price]])
        p = self.action_prob(sm_id, price)
        self.v[sm_id, self.p_id[price]] = np.dot(p, np.array([self.q[sm_id, self.p_id[price]], 0])) + self.alpha * entropy(p)
