import sys

sys.path.append("src")
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data_analysis import DataProcessor
from analysis.plot_time_evolution import Y, compute_ma

states = [1, 2, 3, 4]
subject_ids = [1, 2, 3, 4, 5]
ys = [Y.CA, Y.RT, Y.R]
win_size = 15
win_std = 4

################################
# Current 2AFC vs closest PMT
################################

# reaction time

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()
    for y in ys:
        df = compute_ma(df, y, win_size, win_std)

    for state, ax in zip(states, axs):
        c0 = df["bonus_trial"] == False
        c1 = df["bonus_trial"] == True
        c2 = (df["Session type"] == "testing") & (df["Session ID"] > 0)
        c3 = df["State"] == state

        b = df.loc[c1 & c2 & c3].reset_index(drop=False)

        x_last_trial = df.loc[b["index"] - 1, "reaction_time"]
        x_current_trial = df.loc[b["index"], "reaction_time"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(
            x=x_last_trial, y=x_current_trial, scatter_kws={"alpha": 0.3, "s": 5}, ax=ax
        )
        ax.set_xlabel("")
        ax.set_xlim([0, 3])
        ax.set_ylabel("")
        ax.set_ylim([0, 3])

        if subject_id == 1:
            ax.set_title(f"State {state}")

        if state == 1:
            ax.set_ylabel(f"Subject {subject_id}")

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)

fig.suptitle("Regular (2AFC) vs adjacent PMT trial", fontsize=13, y=0.96)
plt.title("Reaction time", pad=20, fontsize=12)
plt.xlabel("Reaction time (s) for current (regular/2AFC) trial", fontsize=12)
plt.ylabel(
    "Reaction time (s) for adjacent PMT trial",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/correlations/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}RT_pmt-v-regular_adjacent.svg")
plt.savefig(f"{save_path}RT_pmt-v-regular_adjacent.png")
plt.close()


##########
# choice accuracy

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()
    for y in ys:
        df = compute_ma(df, y, win_size, win_std)

    for state, ax in zip(states, axs):
        c0 = df["bonus_trial"] == False
        c1 = df["bonus_trial"] == True
        c2 = (df["Session type"] == "testing") & (df["Session ID"] > 0)
        c3 = df["State"] == state

        b = df.loc[c1 & c2 & c3].reset_index(drop=False)

        x_last_trial = df.loc[b["index"] - 1, "Choice accuracy (MA difficulty)"]
        x_current_trial = df.loc[b["index"], "Choice accuracy (MA difficulty)"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(
            x=x_last_trial, y=x_current_trial, scatter_kws={"alpha": 0.3, "s": 5}, ax=ax
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        if subject_id == 1:
            ax.set_title(f"State {state}")

        if state == 1:
            ax.set_ylabel(f"Subject {subject_id}")

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)

fig.suptitle("Regular (2AFC) vs adjacent PMT trial", fontsize=13, y=0.96)
plt.title("Choice accuracy", pad=20, fontsize=12)
plt.xlabel("Choice accuracy for current (regular/2AFC) trial", fontsize=12)
plt.ylabel(
    "Choice accuracy for adjacent PMT trial",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/correlations/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}CA_pmt-v-regular_adjacent.svg")
plt.savefig(f"{save_path}CA_pmt-v-regular_adjacent.png")
plt.close()


### workbench
import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()
    for y in ys:
        df = compute_ma(df, y, win_size, win_std)

    for state, ax in zip(states, axs):
        c0 = df["bonus_trial"] == False
        c1 = df["bonus_trial"] == True
        c2 = (df["Session type"] == "testing") & (df["Session ID"] > 0)
        c3 = df["State"] == state

        df_reg = df.loc[c0 & c2 & c3].reset_index(drop=False)
        df_pmt = df.loc[c1 & c2 & c3].reset_index(drop=False)

        closest_ids = [
            find_nearest(np.array(df_pmt["index"]), i) for i in list(df_reg["index"])
        ]

        x = df_reg["reaction_time"]
        y = df.loc[closest_ids, "reaction_time"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.3, "s": 5}, ax=ax)
        ax.set_xlabel("")
        ax.set_xlim([0, 3])
        ax.set_ylabel("")
        ax.set_ylim([0, 3])

        if subject_id == 1:
            ax.set_title(f"State {state}")

        if state == 1:
            ax.set_ylabel(f"Subject {subject_id}")

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)

fig.suptitle("Regular (2AFC) vs closest PMT trial", fontsize=13, y=0.96)
plt.title("Reaction time", pad=20, fontsize=12)
plt.xlabel("Reaction time (s) for current (regular/2AFC) trial", fontsize=12)
plt.ylabel(
    "Reaction time (s) for closest PMT trial",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/correlations/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}RT_pmt-v-regular_closest.svg")
plt.savefig(f"{save_path}RT_pmt-v-regular_closest.png")
plt.close()


### choice accuracy

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()
    for y in ys:
        df = compute_ma(df, y, win_size, win_std)

    for state, ax in zip(states, axs):
        c0 = df["bonus_trial"] == False
        c1 = df["bonus_trial"] == True
        c2 = (df["Session type"] == "testing") & (df["Session ID"] > 0)
        c3 = df["State"] == state

        df_reg = df.loc[c0 & c2 & c3].reset_index(drop=False)
        df_pmt = df.loc[c1 & c2 & c3].reset_index(drop=False)

        closest_ids = [
            find_nearest(np.array(df_pmt["index"]), i) for i in list(df_reg["index"])
        ]

        x = df_reg["Choice accuracy (MA difficulty)"]
        y = df.loc[closest_ids, "Choice accuracy (MA difficulty)"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.3, "s": 5}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

        if subject_id == 1:
            ax.set_title(f"State {state}")

        if state == 1:
            ax.set_ylabel(f"Subject {subject_id}")

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)

fig.suptitle("Regular (2AFC) vs closest PMT trial", fontsize=13, y=0.96)
plt.title("Choice accuracy", pad=20, fontsize=12)
plt.xlabel("Choice accuracy for current (regular/2AFC) trial", fontsize=12)
plt.ylabel(
    "Choice accuracy for closest PMT trial",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/correlations/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}CA_pmt-v-regular_closest.svg")
plt.savefig(f"{save_path}CA_pmt-v-regular_closest.png")
plt.close()
