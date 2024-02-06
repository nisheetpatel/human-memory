from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from definitions import Action, Done, Info, Reward, State
from model.utils import SlotTaskParams, option_choice_set_2afc, option_choice_set_slots


class Environment(ABC):
    @property
    @abstractmethod
    def choice_set(self) -> list[int, int]:
        """List available actions."""

    @abstractmethod
    def reset(self) -> State:
        """Reset environment state for the new episode."""

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, Done, Info]:
        """Step in the environment."""


class FinalEpisodeError(Exception):
    error: str


@dataclass
class SlotMachines(Environment):
    """
    Slot machines task. On each trial, subject is asked whether they
    would like to play a uniquely identifiable slot machine for a price.
    """

    n_states: int = 8 + 1
    n_actions: int = 2
    n_blocks: int = 100
    n_repeats_per_block: int = 3
    p: SlotTaskParams = SlotTaskParams()

    def __post_init__(self) -> None:
        self.option_rewards = self._define_option_rewards()

        # generate episodes
        self._episode = 0
        self._episode_list = self._pregenerate_all_episodes()
        self.n_episodes = len(self._episode_list)
        self._state = self.reset()

    @property
    def _state_distribution(self) -> np.ndarray:
        """Number of times each state is presented in a block."""

        high = np.array([1, self.p.rel_diff, self.p.rel_diff, 1])
        low = np.array([self.p.rel_diff, 1, 1, self.p.rel_diff])
        stakes = np.hstack([low, high, low, high])

        freq = np.repeat(np.array([self.p.rel_freq, 1]), len(stakes) / 2)

        state_distribution = stakes * freq * self.n_repeats_per_block
        return state_distribution

    def _generate_episodes_for_block(self) -> np.ndarray:
        states = np.arange(16, dtype=int)
        episodes = np.repeat(states, self._state_distribution)
        np.random.shuffle(episodes)
        return episodes

    def _pregenerate_all_episodes(self) -> np.ndarray:
        episode_list = np.array([], dtype=int)
        for _ in range(self.n_blocks):
            episode_list = np.append(episode_list, self._generate_episodes_for_block())
        return episode_list

    @property
    def choice_set(self) -> list[int, int]:
        return option_choice_set_slots(self._state)

    def _define_option_rewards(self) -> np.ndarray:
        slot_rewards = np.random.uniform(low=0, high=0, size=4)
        slot_rewards = np.around(slot_rewards * 2) / 2

        delta = np.array(
            [-2 * self.p.delta, -1 * self.p.delta, self.p.delta, 2 * self.p.delta]
        )

        return np.append(slot_rewards, slot_rewards - delta)

    def reward(self, action: Action) -> Reward:
        option_chosen = self.choice_set[action]
        reward = self.option_rewards[option_chosen]

        # stochastic rewards for slot machines
        if action == 0:
            reward += np.random.randn() * self.p.sigma

        return reward

    def is_task_finished(self) -> bool:
        return self._episode == len(self._episode_list)

    def reset(self) -> State:
        """Update state to the next one (pre-generated)."""

        if self.is_task_finished():
            raise FinalEpisodeError("Go home. No more trials left!")

        self._state = self._episode_list[self._episode]

        return self._state

    def step(self, action: Action) -> Tuple[State, Reward, Done, Info]:
        """Step in the environment."""

        if self.is_task_finished():
            print("Ignoring step calls beyond what the environment allows.")

        # next state, reward, termination
        next_state = -1
        reward = self.reward(action)
        done = True
        info = None

        # update episode counter
        self._episode += 1

        return next_state, reward, done, info


@dataclass
class Memory2AFC(Environment):
    """
    2AFC (2 alternative forced-choice) task with 12 options,
        grouped into 4 sets of 3 options each.
    On each trial, one of the four sets is selected with probability
    p(set), and two options are drawn from it uniformly at random.
    Trials are pre-generated and proceed in sequence.
    """

    _state: int = 0
    n_states: int = 37
    n_actions: int = 2
    delta_1: float = 4
    delta_2: float = 1
    delta_pmt: float = 4
    rel_freq: float = 4
    n_training_episodes: int = 510
    n_testing_episodes: int = 1020
    n_bonus_trials_per_option: int = 20

    def __post_init__(self) -> None:
        # initialize states and q-size
        self._states = np.arange(self.n_states)

        # define rewards for "options"
        option_rewards = [
            10 + self.delta_1,
            10,
            10 - self.delta_1,
            10 + self.delta_2,
            10,
            10 - self.delta_2,
            10 + self.delta_1,
            10,
            10 - self.delta_1,
            10 + self.delta_2,
            10,
            10 - self.delta_2,
        ]
        good_bonus_option_rewards = list(np.array(option_rewards) + self.delta_pmt)
        bad_bonus_option_rewards = list(np.array(option_rewards) - self.delta_pmt)
        self.option_rewards = (
            option_rewards + good_bonus_option_rewards + bad_bonus_option_rewards
        )

        # define state distribution (frequency with which they appear)
        self._state_distribution = np.append(
            np.repeat(np.arange(6), self.rel_freq), np.arange(6, 12), axis=0
        )

        # pregenerate episodes
        self._episode = 0
        self._pregenerate_episodes()
        self.n_episodes = len(self._episode_list)

    @property
    def choice_set(self) -> list[int, int]:
        return option_choice_set_2afc(self._state)

    def reward(self, action: Action) -> Reward:
        option_chosen = self.choice_set[action]
        reward = self.option_rewards[option_chosen]

        # stochastic rewards for the regular options
        if option_chosen < 12:
            reward += np.random.randn()

        return reward

    def _generate_episode_sequence(self, n_episodes: int) -> np.ndarray:
        episode_sequence = np.repeat(
            self._state_distribution, n_episodes / len(self._state_distribution)
        )
        np.random.shuffle(episode_sequence)
        return episode_sequence

    def _insert_bonus_episodes(self, test_episodes: np.ndarray) -> np.ndarray:
        """function to insert bonus trials in given sequence"""
        # for each of the twelve options
        for option in range(12):
            # determine and pick relevant trials
            ids = [i for i, _ in enumerate(test_episodes) if test_episodes[i] == option]

            # randomly select n_bonus_trials_per_option/2 for good and bad bonus options
            np.random.shuffle(ids)
            ids_bonus_1 = ids[: int(self.n_bonus_trials_per_option / 2)]
            ids_bonus_2 = ids[
                int(self.n_bonus_trials_per_option / 2) : self.n_bonus_trials_per_option
            ]

            # put them together and sort in reverse order
            # so that insertion does not change the indexing
            ids_bonus = ids_bonus_1 + ids_bonus_2
            ids_bonus.sort(reverse=True)
            ids.sort()

            # insert bonus trials
            for idx in ids_bonus:
                if idx in ids_bonus_1:
                    test_episodes.insert(idx + 1, test_episodes[idx] + 12)
                elif idx in ids_bonus_2:
                    test_episodes.insert(idx + 1, test_episodes[idx] + 24)

        return np.array(test_episodes)

    def _pregenerate_episodes(self) -> None:
        # Episodes
        training_episodes = self._generate_episode_sequence(self.n_training_episodes)
        test_episodes = self._generate_episode_sequence(self.n_testing_episodes)
        test_episodes = self._insert_bonus_episodes(list(test_episodes))
        self._episode_list = np.append(training_episodes, test_episodes)

    def is_task_finished(self) -> bool:
        return self._episode == len(self._episode_list)

    def reset(self) -> State:
        """Update state to the next one (pre-generated)."""

        if self.is_task_finished():
            raise FinalEpisodeError("Go home. No more trials left!")

        self._state = self._episode_list[self._episode]

        return self._state

    def step(self, action: Action) -> Tuple[State, Reward, Done, Info]:
        """Step in the environment."""

        if self.is_task_finished():
            print("Ignoring step calls beyond what the environment allows.")

        # next state, reward, termination
        next_state = -1
        reward = self.reward(action)
        done = True
        info = None

        # update episode counter
        self._episode += 1

        return next_state, reward, done, info
