import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import stan


class Plotter:
    def __init__(self, fit: stan.fit.Fit, save_path: str) -> None:
        self.fit = fit
        self.n_subjects, self.n_slots, self.n_samples = self.fit["beta"].shape
        self.save_path = save_path

    def _get_df_for_posterior_plots(self) -> pd.DataFrame:
        betas = self.fit["beta"]
        slot_machine_ids = np.tile(
            np.repeat(np.arange(1, self.n_slots + 1), self.n_samples), self.n_subjects
        )
        participant_ids = np.repeat(
            np.arange(1, self.n_subjects + 1), self.n_slots * self.n_samples
        )
        return pd.DataFrame(
            {
                "id": participant_ids,
                "slot_machine_id": slot_machine_ids,
                "beta": betas.flatten(),
            }
        )

    def plot_posterior_over_betas_individual(
        self, save_name: str, perf: pd.DataFrame = None
    ) -> None:
        n_rows, n_cols = (self.n_subjects + 9) // 10, 10
        df_plot = self._get_df_for_posterior_plots()
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 8))

        for i, ax in enumerate(axs.flatten()[: self.n_subjects]):
            sns.kdeplot(
                data=df_plot[df_plot["id"] == i + 1],
                x="beta",
                hue="slot_machine_id",
                palette="tab10",
                ax=ax,
                legend=False,
            )
            if i < len(axs.flatten()) - 10:
                ax.set_xlabel("")
            if i % 10 != 0:
                ax.set_ylabel("")
            ax.set_yticklabels("")
            if perf is not None:
                ax.set_title(
                    f"{i+1}, {perf.loc[i, 'class']}, {perf.loc[i, 'performance']:.1f}"
                )

        fig.tight_layout()

        plt.savefig(self.save_path + save_name + ".png")
        plt.close()

    def plot_posteriors_at_group_level(self, save_name: str) -> None:
        df_plot = pd.DataFrame(
            {
                "slot_machine_id": np.repeat(np.arange(4) + 1, self.n_samples),
                "beta": np.reshape(self.fit["mu_beta"], -1),
                "alpha": np.reshape(self.fit["mu_alpha"], -1),
            }
        )

        _, axs = plt.subplots(2)
        for x, ax in zip(["alpha", "beta"], axs):
            sns.kdeplot(
                data=df_plot, x=x, hue="slot_machine_id", palette="tab10", ax=ax
            )
            ax.set_title(f"Group-level posterior over {x}")

        plt.savefig(self.save_path + save_name + ".png")
        plt.close()
        plt.close()
