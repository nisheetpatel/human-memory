from typing import Protocol

import numpy as np


class Env(Protocol):
    def step(self, action: bool) -> tuple:
        ...

    def reset(self) -> tuple:
        ...


class SlotMachinesTask:
    def __init__(self, rel_stakes: int = 3, rel_freq: int = 3):
        # necessary variables for initializing state distribution
        low = np.array([1, rel_stakes, rel_stakes, 1])
        high = np.array([rel_stakes, 1, 1, rel_stakes])
        stakes = np.hstack([high, low, high, low])
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
