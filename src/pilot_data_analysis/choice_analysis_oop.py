from extract_data import extract_data
from enum import Enum, auto
from typing import List
import pandas as pd

# type aliases
class X(Enum):
    ABSOLUTE = auto()
    SPECIFIC = auto()


class Y(Enum):
    CA = "Choice accuracy"
    RT = "reaction_time"
    R = "reward"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    def define_session_order(df: pd.DataFrame) -> pd.DataFrame:
        df["Session type"] = pd.Categorical(
            df["Session type"],
            categories=["practice", "training", "testing"],
            ordered=True,
        )
        return df

    def sort_subject_data(df: pd.DataFrame) -> pd.DataFrame:
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

    df = define_session_order(df)
    df = sort_subject_data(df)
    df = add_new_session_column(df)
    df = add_difficulty_column(df)

    return df


def compute_ma(
    df: pd.DataFrame, y: Y, win_size: int = 20, win_std: float = 5
) -> pd.DataFrame:
    def compute(df: pd.DataFrame, rows, label) -> pd.DataFrame:
        df.loc[rows, f"{y.value} (MA {label})"] = (
            df.loc[rows, y.value]
            .rolling(win_size, win_type="gaussian", center=True)
            .mean(std=win_std)
        )
        return df

    for state in df["State"].unique():

        rows = df["State"] == state
        df = compute(df, rows, label="State")

        for d in df["difficulty"].unique():

            rows = (df["State"] == state) & (df["difficulty"] == d)
            df = compute(df, rows, label="difficulty")

    return df, win_size, win_std


# @dataclass
# class Plotter:
#     x: X
#     y: Y
#     dfs: List[pd.DataFrame]

_, _, dfs, dfs_pmt = extract_data()
ys = [Y.CA, Y.RT, Y.R]


for df in dfs:
    df = preprocess_data(df)
    for y in ys:
        df, _, _ = compute_ma(df=df, y=y)


# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")


def plot_choice_curves(dfs: List[pd.DataFrame], y: Y) -> None:

    # defining subplots
    fig, axs = plt.subplots(
        nrows=len(dfs), ncols=1, sharex=True, sharey=True, figsize=(10, 9)
    )
    axs[0].set_title(f"Temporal evolution of {y.value}", fontsize=12)

    # subplot for each subject
    for df, ax in zip(dfs, axs):
        df = preprocess_data(df)
        df, win_size, win_std = compute_ma(df=df, y=y)

        # plot choice accuracy moving average
        g = sns.lineplot(
            data=df,
            x=df.index,
            y=f"{y.value} (MA State)",
            hue="State",
            # style="difficulty",
            ax=ax,
            palette="Spectral",
        )
        if ax is not axs[0]:
            ax.legend([], [], frameon=False)
        else:
            g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("")

        # plot vertical lines as session separators
        new_sess_trial_ids = list(df.loc[df["new-session"] == True].index)
        new_day_trial_id = new_sess_trial_ids[5]

        if y == Y.CA:
            ypos, ymin, ymax = 0.5, 0.25, 1.02
        elif y == Y.R:
            ypos, ymin, ymax = 10, 8, 15
        elif y == Y.RT:
            ypos, ymin, ymax = 1, 0.5, 2

        # ax.vlines(
        #     x=new_sess_trial_ids,
        #     ymin=ymin,
        #     ymax=ymax,
        #     linestyles="dashed",
        #     linewidth=0.8,
        #     colors="gray",
        # )
        # ax.vlines(
        #     x=new_day_trial_id,
        #     ymin=ymin,
        #     ymax=ymax,
        #     linestyles="solid",
        #     linewidth=1.2,
        #     colors="gray",
        # )
        ax.hlines(
            y=ypos,
            xmin=50,
            xmax=180,
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
        f"{y.value} (Gaussian-smoothed with win-size {win_size}, std {win_std})",
        fontsize=11,
    )

    plt.savefig(
        f"./pilot data/figures/time_evolution/PMT_{'_'.join(y.value.split())}_evolution_win-{win_size}-{win_std}.png"
    )
    plt.savefig(
        f"./pilot data/figures/time_evolution/PMT_{'_'.join(y.value.split())}_evolution_win-{win_size}-{win_std}.svg"
    )
    plt.close()


for y in ys:
    plot_choice_curves(dfs_pmt, y)


for y in ys:
    for df in dfs_pmt:
        pass
