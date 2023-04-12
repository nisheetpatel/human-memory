from typing import Protocol

import numpy as np

from model.models import Agent, NoisyQAgent


class LikelihoodComputer(Protocol):
    def compute_log_likelihood(self, data: np.ndarray, model: NoisyQAgent) -> float:
        ...


class NoisyQLikelihoodComputer:
    def compute_log_likelihood(self, data: np.ndarray, model: NoisyQAgent) -> float:
        log_likelihood = 0.0
        for obs in data:
            idx = model.get_index(state=obs[0])
            prob_actions = model.prob_action(idx, obs[4])
            log_likelihood += np.log(prob_actions[obs[1]])

        return log_likelihood


class LikelihoodCalculator:
    def __init__(self, model: Agent, data: np.ndarray):
        self.model = model
        self.data = data
        self.n_trials = len(data)

    def compute_log_likelihood(self, params: dict[str, float]) -> float:
        self.model.p.update(params)
        log_likelihood = 0.0
        for trial in range(self.n_trials):
            (
                state,
                action,
                reward,
                next_state,
                _,  # zeta,
                action_idx,
                _,  # prob_actions,
            ) = self.data[trial]
            prob_action = self.model.act(state)[1][action_idx]
            if action == 0:
                log_likelihood += np.log(prob_action)
            else:
                log_likelihood += np.log(1 - prob_action)
            self.model.observe(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                }
            )
            self.model.update_values(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                }
            )
            self.model.allocate_memory_resources()
        return log_likelihood
