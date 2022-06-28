from typing import List
from extract_data import extract_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def sort_subject_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Session type"] = pd.Categorical(
        df["Session type"], categories=["practice", "training", "testing"], ordered=True
    )
    df.sort_values(
        by=["Day", "Session type", "Session ID"], inplace=True, ignore_index=True
    )
    return df


def add_new_session_column(df: pd.DataFrame) -> pd.DataFrame:
    trial_number_diff = df[".thisN"].diff()
    df["new-session"] = trial_number_diff < 0
    return df


def add_difficulty_column(df: pd.DataFrame) -> pd.DataFrame:
    # state dependent difficulty
    df["difficulty"] = (df["Condition"] % 3) != 1
    return df


def compute_moving_avg(
    df: pd.DataFrame, win_size: int = 20, win_std: float = 10
) -> pd.DataFrame:

    # initialize new column for choice accuracy moving average
    df["Choice accuracy (MA-state)"] = np.nan

    for state in df["State"].unique():

        # compute moving average and set it in place in its column
        row_idx = df["State"] == state

        df.loc[row_idx, "Choice accuracy (MA-state)"] = (
            df.loc[row_idx, "Choice accuracy"]
            .rolling(win_size, win_type="gaussian", center=True)
            .mean(std=win_std)
        )

        for d in df["difficulty"].unique():
            row_idx = (df["State"] == state) & (df["difficulty"] == d)

            df.loc[row_idx, "Choice accuracy (MA-state-difficulty)"] = (
                df.loc[row_idx, "Choice accuracy"]
                .rolling(win_size, win_type="gaussian", center=True)
                .mean(std=win_std)
            )

    return df


def plot_choice_curves(dfs: List[pd.DataFrame], win_size=20, win_std=5) -> None:

    # defining subplots
    fig, axs = plt.subplots(
        nrows=len(dfs), ncols=1, sharex=True, sharey=True, figsize=(10, 10)
    )
    axs[0].set_title("Temporal evolution of choice accuracy", fontsize=12)

    # subplot for each subject
    for df, ax in zip(dfs, axs):
        df = sort_subject_data(df)
        df = add_new_session_column(df)
        df = add_difficulty_column(df)
        df = compute_moving_avg(df, win_size=win_size, win_std=win_std)

        # plot choice accuracy moving average
        g = sns.lineplot(
            data=df,
            x=df.index,
            y="Choice accuracy (MA-state-difficulty)",
            hue="State",
            style="difficulty",
            ax=ax,
            palette="Spectral",
        )
        if ax is not axs[0]:
            ax.legend([], [], frameon=False)
        else:
            g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("")
        ax.set_ylim([0.25, 1.02])

        # plot vertical lines as session separators
        new_sess_trial_ids = list(df.loc[df["new-session"] == True].index)
        new_day_trial_id = new_sess_trial_ids[5]

        ax.vlines(
            x=new_sess_trial_ids,
            ymin=0,
            ymax=1,
            linestyles="dashed",
            linewidth=0.8,
            colors="gray",
        )
        ax.vlines(
            x=new_day_trial_id,
            ymin=0,
            ymax=1,
            linestyles="solid",
            linewidth=1.2,
            colors="gray",
        )
        ax.hlines(
            y=0.5,
            xmin=20,
            xmax=1680,
            linestyles="dashed",
            linewidth=0.5,
            colors="lightgray",
        )

    # add extra figure on top and remove its axes to add common labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    # adding common axis labels
    plt.xlabel("Trial number", fontsize=11)
    plt.ylabel(
        f"Choice accuracy (Gaussian-smoothed with win-size {win_size}, std {win_std})",
        fontsize=11,
    )

    plt.savefig(
        f"./pilot data/figures/time_evolution/choice_accuracy_evolution_all-subjects_win-{win_size}-{win_std}_difficulty-split.svg"
    )
    plt.close()


def main():
    _, _, dfs = extract_data()
    plot_choice_curves(dfs)


if __name__ == "__main__":
    main()
