from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.data_analysis import DataProcessor

states = [1, 2, 3, 4]
subject_ids = [1, 2, 3, 4, 5]


class Y(Enum):
    CA = "Choice accuracy"
    RT = "reaction_time"
    R = "reward"


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

        # c1 = df["bonus_trial"] == False
        c2 = df["State"] == state
        # c3 = df["reaction_time"] < 3.5

        df = compute(df, (c2), label="State")

        for d in df["difficulty"].unique():

            c4 = df["difficulty"] == d
            rows = c2 & c4
            df = compute(df, rows, label="difficulty")

    return df


### Plotting choice curves
def main():
    ys = [Y.CA, Y.RT, Y.R]
    win_size = 15
    win_std = 4

    for y in ys:
        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(9, 9))

        for subject_id, ax in zip(subject_ids, axs):
            # define data processor and extract subject data
            data_processor = DataProcessor(subject_id=subject_id)
            df = data_processor.extract()
            df = data_processor.preprocess()
            df = compute_ma(df, y, win_size, win_std)

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
                g.legend(loc="center left", title="State", bbox_to_anchor=(1, 0.5))
            ax.set_ylabel("")

            # plot vertical lines as session separators
            new_sess_trial_ids = list(df.loc[df["new_session"] == True].index)
            new_day_trial_id = list(df.loc[df["new_day"] == True].index)

            if y == Y.CA:
                ypos, ymin, ymax = 0.5, 0.25, 1.02
            elif y == Y.R:
                ypos, ymin, ymax = 10, 8, 15
            elif y == Y.RT:
                ypos, ymin, ymax = 1, 0.5, 1.8

            ax.vlines(
                x=new_sess_trial_ids,
                ymin=ymin,
                ymax=ymax,
                linestyles="dashed",
                linewidth=0.8,
                colors="gray",
            )
            ax.vlines(
                x=new_day_trial_id,
                ymin=ymin,
                ymax=ymax,
                linestyles="solid",
                linewidth=1.2,
                colors="gray",
            )
            ax.hlines(
                y=ypos,
                xmin=20,
                xmax=1700,
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
        plt.xlabel("Trial number", fontsize=12)
        plt.ylabel(
            f"{y.value} (Gaussian-smoothed with win-size {win_size}, std {win_std})",
            fontsize=12,
            labelpad=10,
        )
        plt.title(f"Evolution of {y.value} (regular trials only)", pad=20, fontsize=12)

        save_path = "./figures/pilot_data_figures/time_evolution/"

        plt.savefig(f"{save_path}{y.value}_regular.svg")
        plt.savefig(f"{save_path}{y.value}_regular.png")
        plt.close()


if __name__ == "__main__":
    main()
