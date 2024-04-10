from typing import Protocol

import numpy as np


class Env(Protocol):
    def step(self, action: bool) -> tuple:
        ...

    def reset(self) -> tuple:
        ...


class SlotMachinesTask:
    def __init__(self, rel_stakes: int = 3, rel_freq: int = 3):
        # define necessary variables for initial state distribution
        low = np.array([1, rel_stakes, rel_stakes, 1])
        high = np.array([rel_stakes, 1, 1, rel_stakes])
        stakes = np.hstack([high, low, high, low])
        freq = np.repeat(np.array([rel_freq, 1]), len(stakes) / 2)

        # define the initial state distribution
        state_distribution = stakes * freq
        self.state_distribution = state_distribution / np.sum(state_distribution)

        # define other task parameters
        self.delta = 0.75
        self.prices = np.array([-2, -1, 1, 2]) * self.delta

        # set initial state
        self._state = None

    def step(self, action: bool):
        # define observation
        sm_id = self._state // 4
        price = self.prices[self._state % 4]

        # define observed return
        rtrn = np.random.normal(0, 1)

        # define reward
        if action == 0: # Yes
            reward = rtrn - price
        elif action == 1: # No
            reward = 0
        else:
            raise ValueError(action, f"Invalid action {action}. Must be 0 or 1.")

        # define next state, termination, info (feedback observations)
        next_state = -1
        done = True
        info = {"sm_id": sm_id, "return": rtrn, "price": price}

        # reset internal state
        self.reset()

        return next_state, reward, done, info

    def reset(self) -> tuple[int, float]:
        self._state = np.random.choice(np.arange(16), p=self.state_distribution)
        return self._state // 4, self.prices[self._state % 4]
