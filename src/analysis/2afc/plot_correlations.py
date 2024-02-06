from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns

from analysis.data_analysis import DataProcessor
from analysis.plot_time_evolution import Y, compute_ma

sns.set(style="whitegrid", font_scale=1.6)

ys = [Y.CA, Y.RT, Y.R]
win_size = 15
win_std = 4


def extract_all_subjects_data() -> pd.DataFrame:
    subject_ids = [1, 2, 3, 4, 5]

    dfs_list = []

    for subject_id in subject_ids:
        # define data processor and extract subject data
        data_processor = DataProcessor(subject_id=subject_id)
        df = data_processor.extract()
        df = data_processor.preprocess()

        for y in ys:
            df = compute_ma(df, y, win_size, win_std)

        dfs_list += [df]

    return pd.concat(dfs_list)


@dataclass
class RTvCAPlotter:
    df: pd.DataFrame
    x: str = "Choice accuracy (MA difficulty)"
    y: str = "reaction_time"

    def __post_init__(self) -> None:
        self.c_test = self.df["Session type"] == "testing"
        self.info_list = [
            self.reg_trials_info,
            self.pmt_trials_info,
            self.all_trials_info,
        ]

    @property
    def reg_trials_info(self) -> Dict:
        c_reg = self.df["bonus_trial"] == False
        d = {"rows": c_reg & self.c_test, "name": "regular"}
        return d

    @property
    def pmt_trials_info(self) -> Dict:
        c_pmt = (self.df["bonus_trial"] == True) & (self.df["Session ID"] > 0)
        d = {"rows": c_pmt & self.c_test, "name": "PMT"}
        return d

    @property
    def all_trials_info(self) -> Dict:
        c_all = self.df["Session ID"] > 0
        d = {"rows": c_all & self.c_test, "name": "all"}
        return d

    def get_df_to_plot(self, info: Dict) -> pd.DataFrame:
        return self.df.loc[info["rows"]].reset_index(drop=True)

    def annotate_r2(self):
        r, _ = sp.stats.pearsonr(self.df[self.x], self.df[self.y])
        ax = plt.gca()
        ax.text(0.05, 0.8, f"r = {round(r,2)}", transform=ax.transAxes, fontsize=12)

    def plot(self, df_plot: pd.DataFrame, annotate_r2=False):
        g = sns.lmplot(
            data=df_plot,
            x=self.x,
            y=self.y,
            row="Subject ID",
            col="State",
            hue="difficulty",
            x_jitter=0.1,
            scatter=True,
            fit_reg=True,
            facet_kws={"legend_out": True},
            scatter_kws={"alpha": 0.5, "s": 5},
        )
        g.set_axis_labels("Choice accuracy", "Reaction time (s)").set(
            ylim=[0.2, 2.5], xticks=[0, 1], yticks=[0.5, 1.5, 2.5]
        ).fig.subplots_adjust(wspace=0.02)
        if annotate_r2:
            g.map_dataframe(self.annotate_r2)
        return g

    def save_plot(self, g, info: Dict) -> None:
        save_path = "./figures/pilot_data_figures/correlations/"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        save_name = f"RT_vs_CA_{info['name']}-trials"

        # add title
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(f"{info['name']} trials")

        plt.savefig(f"{save_path}{save_name}.svg")
        plt.savefig(f"{save_path}{save_name}.png")
        plt.close()


def main() -> None:
    df = extract_all_subjects_data()
    plotter = RTvCAPlotter(df=df)

    for info in plotter.info_list:
        df_plot = plotter.get_df_to_plot(info)
        g = plotter.plot(df_plot)
        plotter.save_plot(g, info)

    return


if __name__ == "__main__":
    import sys

    sys.path.append("src")
    main()
