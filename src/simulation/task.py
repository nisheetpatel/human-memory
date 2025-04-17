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
        if action not in [0, 1]:
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")
        
        # define observation
        sm_id = self._state // 4
        price = self.prices[self._state % 4]

        # define observed return
        rtrn = np.random.normal(0, 1)

        # Calculate reward based on action (0 is Yes, 1 is No)
        reward = rtrn - price if action == 0 else 0

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


class SlotMachinesTaskWithPredefinedData:
    def __init__(self, predefined_data: list[tuple[int, float, float]]):
        """
        Initialize the task with predefined data.
        
        Args:
            predefined_data (List[Tuple[int, float, float]]): List of tuples containing
                (state, price, rtrn) for each trial.
        """
        self.predefined_data = predefined_data
        self.current_trial = 0
        self.total_trials = len(predefined_data)

    def step(self, action: bool) -> tuple[int, float, bool, dict]:
        if self.current_trial >= self.total_trials:
            raise ValueError("All predefined data has been used!")
        
        if action not in [0, 1]:
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")
        
        state, price, rtrn = self.predefined_data[self.current_trial]

        # Calculate reward based on action (0 is Yes, 1 is No)
        reward = rtrn - price if action == 0 else 0
        
        # define next state, termination, info (feedback observations)
        next_state = -1
        done = True
        info = {"sm_id": state // 4, "return": rtrn, "price": price}

        # increase trial counter
        self.current_trial += 1

        return next_state, reward, done, info

    def reset(self) -> tuple[int, float]:        
        try:
            state, price, _ = self.predefined_data[self.current_trial]
            return state // 4, price
        except IndexError:
            print(f"Reached final trial {self.current_trial}")
            return 0, 0
