# import matplotlib.pyplot as plt
import pandas as pd

# import seaborn as sns
from src.plotting.plotting_utils import plot_logit_preds
from src.simulation.experiment import Experiment
from src.simulation.models import (
    DRA,
    EqualRA,
    FreqRA,
    MaxEntRL,
    StakesRA,
)


def main():
    dfs_choice = []
    for model_class in [DRA, FreqRA, StakesRA, EqualRA, MaxEntRL]:
        exp = Experiment(model_class=model_class, n_params=100, n_episodes=1_000)
        exp.run()
        dfs_choice += [exp.extract_choice_data()]
    df_choice = pd.concat(dfs_choice)

    plot_logit_preds(df_choice, save_name="model-preds.png")

    # # sns.set(font_scale=2)
    # fig = sns.relplot(
    #     data=df_choice,
    #     x="price",
    #     y="action",
    #     hue="sm_id",
    #     style="sm_id",
    #     col="Model",
    #     kind="line",
    #     palette=sns.color_palette(n_colors=4),
    # )
    # fig.set(xticks=df_choice["price"].unique())
    # # plt.show()
    # plt.savefig("./model-predictions.png")
    # plt.close()


if __name__ == "__main__":
    main()