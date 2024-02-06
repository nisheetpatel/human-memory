from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data_analysis import DataProcessor

states = [1, 2, 3, 4]
subject_ids = [1, 2, 3, 4, 5]

#################################################
# Reaction time

#############################
# Regular with last Regular
#############################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == False
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "reaction_time"].reset_index(drop=True)

        x_last_trial = x[:-1]
        x_current_trial = x[1:]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(x=x_last_trial, y=x_current_trial, ax=ax)
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
fig.suptitle("Memory effects for reaction time", fontsize=13, y=0.96)
plt.title("Regular trials", pad=20, fontsize=12)
plt.xlabel("Reaction time (s) for current (regular) trial", fontsize=12)
plt.ylabel(
    "Reaction time (s) for previous (regular) trial with same state",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/memory_effects/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}rt_regular.svg")
plt.savefig(f"{save_path}rt_regular.png")
plt.close()


#############################
# PMT with last PMT
#############################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == True
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "reaction_time"].reset_index(drop=True)

        x_last_trial = x[:-1]
        x_current_trial = x[1:]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(x=x_last_trial, y=x_current_trial, ax=ax)
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
fig.suptitle("Memory effects for reaction time", fontsize=13, y=0.96)
plt.title("PMT trials", pad=20, fontsize=13)
plt.xlabel("Reaction time (s) for current PMT trial", fontsize=12)
plt.ylabel(
    "Reaction time (s) for previous PMT trial with same state", labelpad=20, fontsize=12
)

save_path = "./figures/pilot_data_figures/memory_effects/"
Path(save_path).mkdir(parents=True, exist_ok=True)

# plt.show()

##
plt.savefig(f"{save_path}rt_pmt.svg")
plt.savefig(f"{save_path}rt_pmt.png")
plt.close()


###################################
# PMT with previous (regular) trial
###################################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == True
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "reaction_time"].reset_index(drop=False)
        x_current_trial = x["reaction_time"]
        x_last_trial = df.loc[x["index"] - 1, "reaction_time"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(x=x_last_trial, y=x_current_trial, ax=ax)
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
fig.suptitle("Memory effects for reaction time", fontsize=13, y=0.96)
plt.title("PMT trials", pad=20, fontsize=12)
plt.xlabel("Reaction time (s) for current PMT trial", fontsize=12)
plt.ylabel(
    "Reaction time (s) for previous trial (regular trial with same state)",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/memory_effects/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}rt_pmt_w_prev.svg")
plt.savefig(f"{save_path}rt_pmt_w_prev.png")
plt.close()


###################################################
## choice accuracy

#############################
# Regular with last Regular
#############################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == False
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "Choice accuracy"].reset_index(drop=True)

        x_last_trial = x[:-1]
        x_current_trial = x[1:]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(
            x=x_last_trial, y=x_current_trial, ax=ax, x_jitter=0.2, y_jitter=0.2
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
fig.suptitle("Memory effects for choice accuracy", fontsize=13, y=0.96)
plt.title("Regular trials", pad=20, fontsize=12)
plt.xlabel("Response (correct) for current (regular) trial", fontsize=12)
plt.ylabel(
    "Response (correct) for last (regular) trial with same state",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/memory_effects/"
Path(save_path).mkdir(parents=True, exist_ok=True)

plt.savefig(f"{save_path}ca_regular.svg")
plt.savefig(f"{save_path}ca_regular.png")
plt.close()


#############################
# PMT with last PMT
#############################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == True
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "Choice accuracy"].reset_index(drop=True)

        x_last_trial = x[:-1]
        x_current_trial = x[1:]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(
            x=x_last_trial, y=x_current_trial, ax=ax, x_jitter=0.2, y_jitter=0.2
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
fig.suptitle("Memory effects for choice accuracy", fontsize=13, y=0.96)
plt.title("PMT trials", pad=20, fontsize=13)
plt.xlabel("Response (correct) for current PMT trial", fontsize=12)
plt.ylabel(
    "Response (correct) for previous PMT trial with same state",
    labelpad=20,
    fontsize=12,
)

save_path = "./figures/pilot_data_figures/memory_effects/"
Path(save_path).mkdir(parents=True, exist_ok=True)

# plt.show()

##
plt.savefig(f"{save_path}ca_pmt.svg")
plt.savefig(f"{save_path}ca_pmt.png")
plt.close()


###################################
# PMT with previous (regular) trial
###################################

fig, axs_list = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(9, 9))

for subject_id, axs in zip(subject_ids, axs_list):
    # define data processor and extract subject data
    data_processor = DataProcessor(subject_id=subject_id)
    df = data_processor.extract()
    df = data_processor.preprocess()

    for state, ax in zip(states, axs):
        c1 = df["bonus_trial"] == True
        c2 = df["Session type"] == "testing"
        c3 = df["State"] == state
        c4 = df["reaction_time"] < 3.5

        x = df.loc[c1 & c2 & c3 & c4, "Choice accuracy"].reset_index(drop=False)
        x_current_trial = x["Choice accuracy"]
        x_last_trial = df.loc[x["index"] - 1, "Choice accuracy"]

        # jitter(ax=ax, x=x_last_trial, y=x_current_trial)
        sns.regplot(
            x=x_last_trial, y=x_current_trial, ax=ax, x_jitter=0.2, y_jitter=0.2
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
fig.suptitle("Memory effects for choice accuracy", fontsize=13, y=0.96)
plt.title("PMT trials", pad=20, fontsize=12)
plt.xlabel("Response (correct) for current PMT trial", fontsize=12)
plt.ylabel(
    "Response (correct) for previous trial (regular trial with same state)",
    labelpad=20,
    fontsize=12,
)

plt.savefig(f"{save_path}ca_pmt_w_prev.svg")
plt.savefig(f"{save_path}ca_pmt_w_prev.png")
plt.close()
