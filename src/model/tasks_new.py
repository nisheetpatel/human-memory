import numpy as np


class Task:
    def __init__(self, rel_stakes: int = 3, rel_freq: int = 3):
        self.rel_stakes = rel_stakes
        self.rel_freq = rel_freq

        # necessary variables for initializing state distribution
        self.low_stakes = np.array([1, rel_stakes, rel_stakes, 1])
        self.high_stakes = np.array([rel_stakes, 1, 1, rel_stakes])
        self.stakes = np.hstack(
            [self.high_stakes, self.low_stakes, self.high_stakes, self.low_stakes]
        )
        self.freq = np.repeat(np.array([rel_freq, rel_freq, 1, 1]), 4)

        # defining state distribution
        state_distribution = self.stakes * self.freq
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
    
    def switch_freq_2_and_3(self) -> None:
        # necessary variables for initializing state distribution
        self.freq = np.repeat(np.array([self.rel_freq, 1, self.rel_freq, 1]), 4)

        # defining state distribution
        state_distribution = self.stakes * self.freq
        self.state_distribution = state_distribution / np.sum(state_distribution)

    def switch_freq_reverse_all(self) -> None:
        # necessary variables for initializing state distribution
        self.freq = np.repeat(np.array([1, 1, self.rel_freq, self.rel_freq]), 4)

        # defining state distribution
        state_distribution = self.stakes * self.freq
        self.state_distribution = state_distribution / np.sum(state_distribution)

    def switch_stakes_2_and_3(self) -> None:
        # necessary variables for initializing state distribution
        self.stakes = np.hstack(
            [self.high_stakes, self.high_stakes, self.low_stakes, self.low_stakes]
        )

        # defining state distribution
        state_distribution = self.stakes * self.freq
        self.state_distribution = state_distribution / np.sum(state_distribution)

    def switch_stakes_reverse_all(self) -> None:
        # necessary variables for initializing state distribution
        self.stakes = np.hstack(
            [self.low_stakes, self.high_stakes, self.low_stakes, self.high_stakes]
        )

        # defining state distribution
        state_distribution = self.stakes * self.freq
        self.state_distribution = state_distribution / np.sum(state_distribution)
    