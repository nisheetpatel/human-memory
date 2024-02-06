import numpy as np
from scipy.stats import t, norm
from typing import Callable
from functools import partial


class GaussianInverseGamma:
    def __init__(self, mu_0, kappa_0, alpha_0, beta_0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

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
    def __init__(self, mu_0, kappa_0, alpha_0, beta_0, lambda_val):
        super().__init__(mu_0, kappa_0, alpha_0, beta_0)
        self.lambda_val = lambda_val

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


# # Example Usage of Gaussian Inverse Gamma:
# gig = GaussianInverseGamma(mu_0=0, kappa_0=1, alpha_0=1, beta_0=1)
# data = np.random.normal(5, 2, 100)  # Generate some sample data
# gig.update(data)
# print(gig.get_params())

# # Creating 4 slot machines:
# slot_machines = [
#     GaussianInverseGamma(mu_0=0, kappa_0=1, alpha_0=1, beta_0=1) for _ in range(4)
# ]

# # Update the ith machine with new data:
# data_for_ith_machine = np.random.normal(5, 2, 100)
# slot_machines[2].update(data_for_ith_machine)

# # Example of using the policy functions:
# p = 4.5
# beta = 1.0
# chosen_policy = partial(softmax_choice, beta=beta)
# decision = (
#     chosen_policy(slot_machines[2], p)
#     if chosen_policy == softmax_choice
#     else chosen_policy(slot_machines[2], p)
# )
# print(decision)

#############################################


class TaskBIO:
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
        self.delta = 0.75
        self.reward_noise = 2
        self.prices = np.array([-2, -1, 1, 2]) * self.delta

        # initial state
        self._state = None

    def step(self, action: bool):
        # define observation
        sm_id = self._state // 4
        price = self.prices[self._state % 4]

        # define observed reward
        reward_observed = np.random.normal(-price, self.reward_noise)

        # define next state, termination, info (observed sm_id & price)
        next_state = -1
        done = True
        info = sm_id, price

        # reset internal state
        self.reset()

        return next_state, reward_observed, done, info

    def reset(self) -> tuple[int, float]:
        self._state = np.random.choice(np.arange(16), p=self.state_distribution)
        return self._state // 4, self.prices[self._state % 4]


beta = 2.0
chosen_policy = partial(softmax_choice, beta=beta)


class BayesianIdealObserver:
    def __init__(
        self,
        model=GaussianInverseGamma,
        params: dict = {"mu_0": 0, "kappa_0": 1, "alpha_0": 1, "beta_0": 1},
        policy=chosen_policy,
    ) -> None:
        self.slot_machines = [model(**params) for _ in range(4)]
        self.policy = policy
        self.return_history = [[], [], [], []]

    def act(self, sm_id: int, price: float) -> int:
        return self.policy(self.slot_machines[sm_id], price)

    def update(self, sm_id: int, price: float, reward: float, action: int) -> None:
        # append current return
        self.return_history[sm_id].append(reward)

        # update parameters of currently shown slot machine
        self.slot_machines[sm_id].update(self.return_history[sm_id])


from model.model_new import Simulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

policies = [optimal_choice, softmax_choice, prob_from_t, prob_from_gaussian]

env = TaskBIO()
n_episodes = 18
n_runs = 1000

fig, axs = plt.subplots(1, len(policies), figsize=(16, 5))

# For each policy, simulate multiple runs and accumulate results
for policy_idx, policy in enumerate(policies):
    accumulated_data = []

    # Running the simulation multiple times and accumulating data
    for run in range(n_runs):
        agent = BayesianIdealObserver(
            model=GaussianInverseGamma,
            params={
                "mu_0": 0,
                "kappa_0": 0.01,
                "alpha_0": 1.01,
                "beta_0": 10,
            },  # "lambda_val": 0.2},
            policy=policy,
        )
        simulator = Simulator(env=env, agent=agent, n_episodes=n_episodes)
        data = simulator.train_agent(record_data=True)
        accumulated_data.extend(data)

    df = pd.DataFrame(accumulated_data, columns=["sm_id", "price", "reward", "action"])

    # Define columns to plot
    df["expected_reward"] = -1 * df["price"]
    df["response_yes"] = 1 - df["action"]

    # Plot psychometric curves with averages
    ax = axs[policy_idx]
    sns.lineplot(
        data=df,
        x="expected_reward",
        y="response_yes",
        hue="sm_id",
        palette="tab10",
        ax=ax,
    )
    ax.set_title(policy.__name__)

plt.show()
