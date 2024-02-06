from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from scipy.stats import norm


class Task:
    def __init__(self, rel_stakes: int = 3, rel_freq: int = 3):
        # necessary variables for initializing state distribution
        high = np.array([1, rel_stakes, rel_stakes, 1])
        low = np.array([rel_stakes, 1, 1, rel_stakes])
        stakes = np.hstack([low, high, low, high])
        freq = np.repeat(np.array([rel_freq, 1]), len(stakes) / 2)

        # defining state distribution
        state_distribution = stakes * freq
        self.state_distribution = state_distribution / np.sum(state_distribution)

        # defining other task parameters
        self.delta = 1.
        self.prices = np.array([-2, -1, 1, 2]) * self.delta

        # initial state
        self._state = None

    def step(self, action: bool):
        # define observation
        sm_id = self._state // 4
        price = self.prices[self._state % 4]

        # define observed reward
        reward = 0

        if action == 0: # Yes
            reward = np.random.normal(-price, 0.1)

        # define next state, termination, info (observed sm_id & price)
        next_state = -1
        done = True
        info = sm_id, price

        # reset internal state
        self.reset()

        return next_state, reward, done, info

    def reset(self) -> tuple[int, float]:
        self._state = np.random.choice(np.arange(16), p=self.state_distribution)
        return self._state // 4, self.prices[self._state % 4]


class Agent(Protocol):
    def action_prob(self, sm_id: int, price: float) -> tuple:
        ...

    def act(self, sm_id: int, price: float) -> int:
        ...

    def update(self, sm_id: int, price: float, reward: float, action: int) -> None:
        ...


class DDRA:
    def __init__(self, lr_v: float = 0.05, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_base: float = 5) -> None:
        # define parameters
        self.lr_v = lr_v
        self.lr_s = lr_s
        self.lmda = lmda
        self.sigma_base = sigma_base
        
        # define initial value and noise
        self.v = np.array([0.,0.,0.,0.])
        self.sigma = np.array([1.,1.,1.,1.]) * sigma_base
        self.sigma_history = [self.sigma]

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


class DOtherRA(ABC):
    def __init__(self, lr_v: float = 0.05, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_base: float = 5) -> None:

        self.v = np.array([0,0,0,0])
        self.sigma_scalar = sigma_base
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


class DEqualRA(DOtherRA):
    @property
    def norm(self):
        norm_factor = np.ones(4)
        return 4 * norm_factor / np.sum(norm_factor[:4])


class DFreqRA(DOtherRA):
    @property
    def norm(self):
        norm_factor = np.array([np.sqrt(3),np.sqrt(3),1,1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


class DStakesRA(DOtherRA):
    @property
    def norm(self):
        norm_factor = np.array([np.sqrt(3),1,np.sqrt(3),1])
        return 4 * norm_factor / np.sum(norm_factor[:4])


class Simulator:

    def __init__(self, env: Task, agent: DDRA, n_episodes: int = 1_000):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def run_episode(self) -> tuple[int, float, float, bool]:
        sm_id, price = self.env.reset()
        action = self.agent.act(sm_id, price)
        _, reward, _, _ = self.env.step(action)
        self.agent.update(sm_id, price, reward, action)
        return (sm_id, price, reward, action)

    def train_agent(self, record_data: bool = False) -> list[tuple]:
        data = []
        for _ in range(self.n_episodes):
            data_tuple = self.run_episode()
            if record_data:
                data.append(data_tuple)
        return data