from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Protocol

import numpy as np
from scipy.stats import norm, t


class Agent(Protocol):
    def action_prob(self, sm_id: int, price: float) -> tuple:
        ...

    def act(self, sm_id: int, price: float) -> int:
        ...

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
        ...


######################################################################
# Resource Allocation Models
######################################################################

class DRA:
    def __init__(self, lr_v: float = 0.01, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_base: float = 5) -> None:
        # define parameters
        self.lr_v = lr_v
        self.lr_s = lr_s
        self.lmda = lmda
        self.sigma_base = sigma_base
        
        # define initial value and noise
        self.v = np.array([0.,0.,0.,0.])
        self.sigma = np.array([1.,1.,1.,1.]) * sigma_base / 2
        self.sigma_history = [sigma_base / 2]

    def action_prob(self, sm_id: int, price: float):
        p_no = norm.cdf((price - self.v[sm_id]) / self.sigma[sm_id])
        # p_no = (np.random.normal(self.v[sm_id], self.sigma[sm_id]) - price) < 0
        return (1-p_no, p_no)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def _compute_grad_noise(self, sm_id: int, price: float, reward: float, action: int):
        grad_cost = self.lmda * (self.sigma / self.sigma_base ** 2 - 1 / self.sigma)
        grad_reward = 0

        if action == 0:
            self.v[sm_id] += self.lr_v * (reward - self.v[sm_id])

            x = (price - self.v[sm_id]) / self.sigma[sm_id]
            grad_reward = norm.pdf(x) / (1 - norm.cdf(x)+ 1e-4) * x / self.sigma[sm_id]
            grad_reward *= reward
        
        grad = -grad_cost
        grad[sm_id] += grad_reward

        return grad

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
        # update values
        self.v[sm_id] += self.lr_v * (rtrn - self.v[sm_id])

        # update noise
        self.sigma += self.lr_s * self._compute_grad_noise(sm_id, price, reward, action)
        self.sigma = np.clip(self.sigma, 0.01, self.sigma_base)


class OtherRA(ABC):
    def __init__(self, lr_v: float = 0.01, lr_s: float = 0.05, lmda: float = 0.1,
                 sigma_base: float = 5) -> None:

        self.v = np.array([0,0,0,0])
        self.sigma_scalar = sigma_base / 2
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
        # p_no = (np.random.normal(self.v[sm_id], self.sigma[sm_id]) - price) < 0
        return (1-p_no, p_no)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
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


######################################################################
# RL models
######################################################################


def softargmax(x: np.ndarray, beta: float = 1) -> np.ndarray:
    y = np.exp(beta * x - np.max(beta * x))
    return y / y.sum()


class RL:
    def __init__(self, lr_v: float = 0.05, **kwargs) -> None:
        # define parameters
        self.lr_v = lr_v

        # define initial values
        self.v = np.array([0., 0., 0., 0.])

    def action_prob(self, sm_id: int, price: float) -> np.ndarray:
        return np.argmax(np.array([self.v[sm_id] - price, 0]))

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
        self.v[sm_id] += self.lr_v * (rtrn - self.v[sm_id])


class MaxEntRL:
    def __init__(self, lr_v: float = 0.05, alpha: float = 0.5) -> None:
        # define parameters
        self.lr_v = lr_v
        self.alpha = alpha

        # initialize values
        self.v = np.array([0.,0.,0.,0.])

    def action_prob(self, sm_id: int, price: float) -> np.ndarray:
        return softargmax(np.array([self.v[sm_id]-price, 0]), 1 / self.alpha)

    def act(self, sm_id: int, price: float):
        return np.random.choice([0,1], p=self.action_prob(sm_id, price))

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
        self.v[sm_id] += self.lr_v * (rtrn - self.v[sm_id])


######################################################################
# Bayesian Ideal Observers
######################################################################

class GaussianInverseGamma:
    def __init__(self, mu_0, kappa_0, alpha_0, beta_0, lambda_val):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.lambda_val = min(lambda_val, 0.9)

    def update(self, data):
        n = len(data)
        x_bar = np.mean(data)

        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / kappa_n

        alpha_n = self.alpha_0 + n / 2
        beta_n = (
            self.beta_0
            + 0.5 * np.sum((data - x_bar) ** 2)
            + (n * self.kappa_0) / (2 * (self.kappa_0 + n)) * (x_bar - self.mu_0) ** 2
        )

        # Update the parameters
        self.mu_0 = mu_n
        self.kappa_0 = kappa_n
        self.alpha_0 = alpha_n
        self.beta_0 = beta_n

    def get_params(self):
        return {
            "mu": self.mu_0,
            "kappa": self.kappa_0,
            "alpha": self.alpha_0,
            "beta": self.beta_0,
        }


class LeakyGaussianInverseGamma(GaussianInverseGamma):

    def update(self, data):
        # Leaky update for the mean
        x_bar = np.mean(data)
        self.mu_0 = (1 - self.lambda_val) * self.mu_0 + self.lambda_val * x_bar

        # Rest of the updates can remain similar to the base class
        # (or can be modified for further "leakiness" if required)
        super().update(data)


# Define the type for our choice policies
ChoicePolicy = Callable[[GaussianInverseGamma, float], int]


def optimal_choice(gig: GaussianInverseGamma, p: float) -> int:
    return 0 if gig.mu_0 > p else 1


def softmax_choice(gig: GaussianInverseGamma, p: float, beta: float = 2) -> int:
    prob_yes = 1 / (1 + np.exp(-beta * (gig.mu_0 - p)))
    return 0 if np.random.rand() < prob_yes else 1


def prob_from_t(gig: GaussianInverseGamma, p: float) -> int:
    scale = np.sqrt(gig.beta_0 * (1 + gig.kappa_0) / (gig.alpha_0 * gig.kappa_0))
    prob_mu_greater_p = 1 - t.cdf(p, 2 * gig.alpha_0, gig.mu_0, scale)
    return 0 if np.random.rand() < prob_mu_greater_p else 1


def prob_from_gaussian(gig: GaussianInverseGamma, p: float) -> int:
    prob_return_greater_p = 1 - norm.cdf(
        p, gig.mu_0, np.sqrt(gig.beta_0 / (gig.alpha_0))
    )
    return 0 if np.random.rand() < prob_return_greater_p else 1


class BayesianIdealObserver:
    def __init__(
        self,
        model: GaussianInverseGamma,
        policy: ChoicePolicy,
        mu_0: float = 0,
        kappa_0: float = 1,
        alpha_0: float = 1,
        beta_0: float = 1,
        lambda_val: float = 0.5,
    ) -> None:
        params = {"mu_0": mu_0, "kappa_0": kappa_0, "alpha_0": alpha_0, "beta_0": beta_0, "lambda_val": lambda_val}
        self.slot_machines = [model(**params) for _ in range(4)]
        self.policy = policy
        self.return_history = []

    def act(self, sm_id: int, price: float) -> int:
        return self.policy(self.slot_machines[sm_id], price)

    def _get_return_history(self, sm_id: int) -> list:
        return [x[1] for x in self.return_history if x[0] == sm_id]

    def update(self, sm_id: int, price: float, reward: float, rtrn: float, action: int):
        # append current return
        self.return_history.append([sm_id, rtrn])

        # update parameters of currently shown slot machine
        self.slot_machines[sm_id].update(self._get_return_history(sm_id))


class ForgetfulBayesianObserver(BayesianIdealObserver):
    def _get_return_history(self, sm_id: int) -> list:
        n_back = 20
        return [x[1] for x in self.return_history[-n_back:] if x[0] == sm_id]


class OptimalBIO(BayesianIdealObserver):
    def __init__(self, model = GaussianInverseGamma, policy = partial(optimal_choice), **kwargs):
        super().__init__(model=model, policy=policy)


class ForgetfulOptimalBIO(ForgetfulBayesianObserver):
    def __init__(self, model = GaussianInverseGamma, policy = partial(optimal_choice), **kwargs):
        super().__init__(model=model, policy=policy)


class SoftmaxBIO(BayesianIdealObserver):
    def __init__(self, model = GaussianInverseGamma, policy = partial(softmax_choice), **kwargs):
        super().__init__(model=model, policy=policy)


class ForgetfulSoftmaxlBIO(ForgetfulBayesianObserver):
    def __init__(self, model = GaussianInverseGamma, policy = partial(softmax_choice), **kwargs):
        super().__init__(model=model, policy=policy)


class ProbTBIO(BayesianIdealObserver):
    def __init__(self, model = GaussianInverseGamma, policy = partial(prob_from_t), **kwargs):
        super().__init__(model=model, policy=policy)


class ForgetfulProbTBIO(ForgetfulBayesianObserver):
    def __init__(self, model = LeakyGaussianInverseGamma, policy = partial(prob_from_t), **kwargs):
        super().__init__(model=model, policy=policy)
