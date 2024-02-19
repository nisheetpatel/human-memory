from time import perf_counter as timer

import numpy as np
import pandas as pd
import psytrack
from matplotlib import pyplot as plt

from src.simulation.models import DRA
from src.simulation.simulator import Simulator
from src.simulation.task import SlotMachinesTask

# initialize env, agents
env = SlotMachinesTask()
agent_classes = [DRA] #, DFreqRA, DStakesRA, DEqualRA]
agents = [agent(lr_s=0.1) for agent in agent_classes]
simulators = [Simulator(env, agent, 192 * 2) for agent in agents]

################################################################
# simply record the sigmas
################################################################
sigmas = []

for simulator in simulators:
    simulator.train_agent()
    sigmas.append(simulator.agent.sigma)

# switch the task appearance frequencies and stakes
env.switch_freq_reverse_all()
env.switch_stakes_reverse_all()

for simulator in simulators:
    simulator.train_agent()
    sigmas.append(simulator.agent.sigma)


################################################################
# definitions for psytrack model
################################################################

# function to transform data for psytrack analyses
def organize_data_for_psytrack(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data for Analyses."""

    data = data.copy()

    for i in range(4):
        data[f"X{i+1}"] = data["price"] * (data["sm_id"] == i)

    # Assuming 'id' is still needed, if not remove it from the list below
    return data.loc[:, ["Model", "Task", "action", "X1", "X2", "X3", "X4"]]


# redefine psytrack model
class PsytrackModel:
    def __init__(self, sigma: float = 0.25, sigma_init: float = 0.5) -> None:
        """
        sigma : required, controls trial-to-trial variability
        sigInit : optional, controls the variability on the first trial 
            (e.g. how close weights must initialize to 0). It is often best 
            to include this hyperparameter and set it to a high value, as you 
            often prefer your data to determine where the weights ought to 
            initialize. Otherwise, sigInit will be set equal to sigma.
        """
        # Define parameters for psytrack optimization
        self.weights = {"bias": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1}
        self.n_weights = sum(self.weights.values())
        self._hyperparams = self._init_hyperparams(sigma_init, sigma)
        self.opt_list = ["sigma"]

    def _init_hyperparams(self, sigma_init, sigma) -> dict:
        return {
            "sigInit": [sigma_init] * self.n_weights,
            "sigma": [sigma] * self.n_weights,
            "sigDay": None,
        }

    @staticmethod
    def _organize_data(data: pd.DataFrame) -> dict:
        """Organize data for psytrack."""
        if "X1" not in data.columns:
            data = organize_data_for_psytrack(data, test_only=False)
        return {
            "inputs": {
                "x1": np.asarray(data["X1"]).reshape(-1, 1),
                "x2": np.asarray(data["X2"]).reshape(-1, 1),
                "x3": np.asarray(data["X3"]).reshape(-1, 1),
                "x4": np.asarray(data["X4"]).reshape(-1, 1),
            },
            "y": np.asarray(data["action"]),
        }

    def fit(self, data: pd.DataFrame, jump: int = 2) -> None:
        """Fit psytrack model to data."""

        print("Fitting psytrack model...")
        start = timer()
        data_dict = self._organize_data(data)
        (
            self.hyperparams,
            self.evidence,
            self.w_mode,
            self.hess_info,
        ) = psytrack.hyperOpt(
            data_dict, self._hyperparams, self.weights, self.opt_list, jump=jump
        )

        print(f"Done in {timer() - start:.2f} seconds.\n")

    def plot_weights(self):
        """Plot psytrack inferred weights."""

        x = np.arange(self.w_mode.shape[1]) + 1
        err = self.hess_info["W_std"]
        colors = ["grey"] + plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

        for i in range(self.n_weights):
            label = f"Slope {i}" if i > 0 else "Bias"
            plt.plot(x, self.w_mode[i], label=label, color=colors[i], zorder=1)
            plt.fill_between(
                x,
                self.w_mode[i] - 1.96 * err[i],
                self.w_mode[i] + 1.96 * err[i],
                alpha=0.2,
                color=colors[i],
                zorder=0,
            )
        plt.xlabel("Trial number")
        plt.ylabel("Psytrack inferred weights")
        plt.axhline(0, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 1, c="black", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 2, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.axvline(192 * 3, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        plt.legend()
        # plt.title(f"Psytrack estimates of slopes for subject {}")
        plt.show()
    
    def get_weights_and_errors(self):
        """Return the mode of the weights and their standard errors."""
        return self.w_mode, self.hess_info['W_std']


################################################################
# Collect data to analyze with psytrack
################################################################

# re-initialize env, agents
env = SlotMachinesTask()
agent = DRA(lr_s=0.1)
simulator = Simulator(env, agent, 192 * 2)

# initialize data columns and data list
data_columns = ["sm_id", "price", "reward", "action"]
data = []

# collect data from each model
data_tuples = simulator.train_agent(record_data=True)
df = pd.DataFrame.from_records(data_tuples, columns=data_columns)
df["Model"] = "DRA"
df["Task"] = "before-switch"
data.append(df)

# switch the task appearance frequencies and stakes
env.switch_freq_reverse_all()
env.switch_stakes_reverse_all()

# collect data after switching task parameters
new_data_tuples = simulator.train_agent(record_data=True)
new_df = pd.DataFrame.from_records(new_data_tuples, columns=data_columns)
new_df["Model"] = "DRA"
new_df["Task"] = "after-switch"
data.append(new_df)

# gather data
df = pd.concat(data, ignore_index=True)
df_clean = organize_data_for_psytrack(df)

# define psytrack model, fit, and plot
psy = PsytrackModel()
psy.fit(df_clean)
psy.plot_weights()


##################################################################
# Plotting multiple runs in subplots
##################################################################

# Number of simulations to run
n_runs = 5
block_len = 192

# Store weights and errors from each run
weights_runs = []
errors_runs = []

# Run simulations and fit the model
for _ in range(n_runs):
    # re-initialize env, agent, simulator for each run
    env = SlotMachinesTask()
    agent = DRA(lr_s=0.1)
    simulator = Simulator(env, agent, block_len * 2)

    # initialize data columns and data list
    data_columns = ["sm_id", "price", "reward", "action"]
    data = []

    # collect data from each model
    data_tuples = simulator.train_agent(record_data=True)
    df = pd.DataFrame.from_records(data_tuples, columns=data_columns)
    df["Model"] = "DRA"
    df["Task"] = "before-switch"
    data.append(df)

    # switch the task appearance frequencies and stakes
    env.switch_freq_reverse_all()
    env.switch_stakes_reverse_all()

    # collect data after switching task parameters
    new_data_tuples = simulator.train_agent(record_data=True)
    new_df = pd.DataFrame.from_records(new_data_tuples, columns=data_columns)
    new_df["Model"] = "DRA"
    new_df["Task"] = "after-switch"
    data.append(new_df)

    # gather data
    df = pd.concat(data, ignore_index=True)
    df_clean = organize_data_for_psytrack(df)
    
    # Fit the model
    psy = PsytrackModel(0.25, 0.5)
    psy.fit(df_clean)
    
    # Store the weights and errors
    weights, errors = psy.get_weights_and_errors()
    weights_runs.append(weights)
    errors_runs.append(errors)

# Now plot the weights from all runs in a single figure
fig, axes = plt.subplots(1, n_runs, figsize=(15, 5), sharey=True)
for ax, weights, errors in zip(axes, weights_runs, errors_runs):
    x = np.arange(weights.shape[1]) + 1
    for i in range(1, weights.shape[0]):
        ax.plot(x, weights[i], label=f'Slope {i}')
        ax.fill_between(x, weights[i] - 1.96 * errors[i], weights[i] + 1.96 * errors[i], alpha=0.2)
        ax.axvline(block_len * 1, c="black", ls="--", lw=1, alpha=0.5, zorder=0)
        ax.axvline(block_len * 2, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
        ax.axvline(block_len * 3, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
    ax.set_xlabel("Trial number")
    if ax == axes[0]:
        ax.set_ylabel("Slope")
    if ax == axes[2]:
        ax.set_title("Psytrack Inferred Slopes")

# Add a legend outside of the last subplot
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()


##################################################################
# Plotting average across multiple runs
##################################################################


import numpy as np
import pandas as pd
import psytrack
from matplotlib import pyplot as plt

from src.simulation.models import DRA
from src.simulation.simulator import Simulator
from src.simulation.task import SlotMachinesTask

# Parameters
block_len = 192
n_simulations = 50  # Number of simulations to run
n_episodes = block_len * 2  # Number of episodes per simulation

# Storage for weights from each run
all_weights = []

# initialize data columns and data list
data_columns = ["sm_id", "price", "reward", "action"]
data = []

for simulation in range(n_simulations):
    # Initialize env, agent and simulator for each run
    env = SlotMachinesTask()
    agent = DRA(lr_s=0.1)
    simulator = Simulator(env, agent, n_episodes)
    
    # Run the simulation and collect data
    df = pd.DataFrame(simulator.train_agent(True), columns=data_columns)
    df["Model"] = "DRA"
    df["Task"] = "before-switch"
    df_clean = organize_data_for_psytrack(df)

    # reverse contingencies in the task
    env.switch_freq_reverse_all()
    env.switch_stakes_reverse_all()

    # Run the simulation again and collect data
    simulator.train_agent(record_data=True)
    df_switch = pd.DataFrame(simulator.train_agent(True), columns=data_columns)
    df_switch["Model"] = "DRA"
    df_switch["Task"] = "after-switch"
    df_clean_switch = organize_data_for_psytrack(df_switch)
    
    # concatenate data
    df = pd.concat([df_clean, df_clean_switch], ignore_index=True)

    # Fit psytrack model
    psy_model = PsytrackModel()
    psy_model.fit(df)
    
    # Store the weights
    all_weights.append(psy_model.w_mode)
    
# Calculate average weights and standard errors across runs
average_weights = np.mean(all_weights, axis=0)
standard_errors = np.std(all_weights, axis=0) / np.sqrt(n_simulations)

# Plot the average weights
x = np.arange(average_weights.shape[1]) + 1
plt.figure(figsize=(10, 5))
for i in range(1, average_weights.shape[0]):
    plt.plot(x, average_weights[i], label=f'Slope {i}')
    plt.fill_between(x, average_weights[i] - 1.96 * standard_errors[i], average_weights[i] + 1.96 * standard_errors[i], alpha=0.2)
plt.axvline(block_len * 1, c="black", ls="--", lw=1, alpha=0.5, zorder=0)
plt.axvline(block_len * 2, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
plt.axvline(block_len * 3, c="grey", ls="--", lw=1, alpha=0.5, zorder=0)
plt.xlabel('Trial number')
plt.ylabel('Average inferred slopes')
plt.title(f'Average Psytrack Inferred Slopes Across N={n_simulations} Simulations')
plt.legend()
plt.show()
