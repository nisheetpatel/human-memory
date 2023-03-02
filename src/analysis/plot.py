import itertools
from dataclasses import asdict, dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_PATH = "./figures/slot-machines/pilot_3/"


@dataclass
class FigureParams:
    """Params for generating plots with seaborn."""

    plot_function: callable
    save_name_suffix: str
    save_name: str = ""
    x: str = "Difficulty"
    y: str = "Response"
    hue: Optional[str] = "Slot Machine ID"
    style: Optional[str] = "Slot Machine ID"

    def parse(self) -> dict:
        """Separates plotting params from plotting function and save name."""
        plot_params = asdict(self)
        plot_function = plot_params.pop("plot_function")
        save_name = plot_params.pop("save_name") + plot_params.pop("save_name_suffix")
        return plot_params, plot_function, save_name


@dataclass
class FigureParamsPooled(FigureParams):
    """Params for generating plots with seaborn."""

    plot_function: callable = sns.lineplot
    save_name_suffix: str = "_pooled"


@dataclass
class FigureParamsIndividual(FigureParams):
    """Additional params for generating plots for individual subjects."""

    kind: str = "line"
    col: str = "participant_id"
    col_wrap: int = 5
    plot_function: callable = sns.relplot
    save_name_suffix: str = "_individual"


@dataclass
class Figure:
    data: pd.DataFrame
    params: dict
    save_name: str
    plot_function: Union[sns.lineplot, sns.relplot]

    def plot(self, save=True) -> None:
        fig = self.plot_function(data=self.data, **self.params)
        self._set_ticks(fig)
        if save:
            plt.savefig(FIG_PATH + self.save_name + ".png")
            plt.close()
        else:
            plt.show()

    def _set_ticks(self, fig) -> None:
        fig.set(xticks=self.data[self.params["x"]].unique())


abbr = {
    "Choice Accuracy": "CA",
    "Response Time": "RT",
    "Response": "Resp",
    "Difficulty": "Diff",
    "Slot Machine ID": "SM",
    "block_type": "Block",
    None: "",
}


@dataclass
class FigureFactory:
    """Figure factory with list of x and y variables to plot."""

    plot_params: list[tuple[str, str, Union[str, None]]]

    def compile_params(self) -> list[FigureParams]:
        """Compiles params for all figures to be generated."""

        fig_params = [FigureParamsPooled, FigureParamsIndividual]
        params = []

        for fig_param, p in itertools.product(fig_params, self.plot_params):
            save_name = f"{abbr[p[1]]}-vs-{abbr[p[0]]}-by-{abbr[p[2]]}"
            params.append(
                fig_param(x=p[0], y=p[1], hue=p[2], style=p[2], save_name=save_name)
            )

        return params

    @staticmethod
    def _make_figure(params: FigureParams, data=pd.DataFrame) -> Figure:
        plot_params, plot_function, save_name = params.parse()
        return Figure(
            data=data,
            params=plot_params,
            save_name=save_name,
            plot_function=plot_function,
        )

    def make_figures(self, data: pd.DataFrame) -> list[Figure]:
        params = self.compile_params()
        return [self._make_figure(param, data) for param in params]


def plot_test_block_figures(df: pd.DataFrame, filter_df=True) -> None:
    if filter_df:
        df = df[df["block_type"] == "test"]

    plots = [("Difficulty", "Response", "Slot Machine ID")]  # (x, y, hue)

    figures = FigureFactory(plots).make_figures(df)
    for figure in figures:
        figure.params["palette"] = "tab10"
        figure.plot()


def plot_all_block_figures(df: pd.DataFrame) -> None:
    plots = [
        # x, y, hue
        ("Slot Machine ID", "Choice Accuracy", "block_type"),
        ("Slot Machine ID", "Response Time", "block_type"),
        ("Difficulty", "Choice Accuracy", "block_type"),
        ("Difficulty", "Response Time", "block_type"),
    ]
    figures = FigureFactory(plots).make_figures(df)
    for figure in figures:
        figure.plot()


if __name__ == "__main__":
    import os

    from .process import main as load_data

    if os.path.exists("../../data/"):
        df_data, _ = load_data("../../data/pilot_slot-machines_3a/")

        plot_test_block_figures(df_data)
        plot_all_block_figures(df_data)

    else:
        print("No data found. Are you in the root directory?")
