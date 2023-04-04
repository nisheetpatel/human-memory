import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit  # pylint: disable=no-name-in-module


def pooled_logit_plots(data: pd.DataFrame) -> None:
    _, ax = plt.subplots()
    data_block = (
        data.loc[
            data["block_id"] > 0,
            [
                "participant_id",
                "state",
                "Slot Machine ID",
                "slot_machine_mean_payoff",
                "price",
                "correct_response",
                "Choice Accuracy",
                "Response Time",
                "Difficulty",
                "Response",
            ],
        ]
        .dropna()
        .copy()
    )

    logit_params = []
    for sm in [0, 1, 2, 3]:
        data_sm = data_block.loc[data_block["Slot Machine ID"] == sm]
        reg = smf.logit(formula="Response ~ Difficulty", data=data_sm).fit(maxiter=1000)
        logit_params.append(list(reg.params))
        # print(f"Slot Machine {sm}")
        # print(reg.summary())
        # print("\n\n\n")

    logit_params = np.asarray(logit_params)
    exp_b0 = np.exp(logit_params[:, 0])
    slopes = logit_params[:, 1] * exp_b0 / (1 + exp_b0) ** 2

    x = np.linspace(-10, 10, 1000)
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    for i, (p, l) in enumerate(zip(logit_params, linestyles)):
        ax.plot(
            x,
            expit(p[1] * x + p[0]),
            label=f"SM {i+1}, slope {slopes[i]:.2f}",
            linestyle=l,
        )

    ax.set_xlim([-10, 10])
    ax.set_xticks([-2, -1, 1, 2])
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Response")
    ax.set_title("Pooled Logisitc Regression fits")
    ax.legend()

    plt.show()


def pooled_logit_plots_per_block(data: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for block_id, ax in zip([0, 1, 2, 3], axs):
        data_block = (
            data.loc[
                data["block_id"] == block_id,
                [
                    "participant_id",
                    "state",
                    "Slot Machine ID",
                    "slot_machine_mean_payoff",
                    "price",
                    "correct_response",
                    "Choice Accuracy",
                    "Response Time",
                    "Difficulty",
                    "Response",
                ],
            ]
            .dropna()
            .copy()
        )

        logit_params = []
        for sm in [0, 1, 2, 3]:
            data_sm = data_block.loc[data_block["Slot Machine ID"] == sm]
            reg = smf.logit(formula="Response ~ Difficulty", data=data_sm).fit(
                maxiter=1000
            )
            logit_params.append(list(reg.params))
            # print(f"Slot Machine {sm}")
            # print(reg.summary())
            # print("\n\n\n")

        logit_params = np.asarray(logit_params)
        exp_b0 = np.exp(logit_params[:, 0])
        slopes = logit_params[:, 1] * exp_b0 / (1 + exp_b0) ** 2

        x = np.linspace(-10, 10, 1000)
        linestyles = ["solid", "dashed", "dotted", "dashdot"]

        for i, (p, l) in enumerate(zip(logit_params, linestyles)):
            ax.plot(
                x,
                expit(p[1] * x + p[0]),
                label=f"SM {i+1}, slope {slopes[i]:.2f}",
                linestyle=l,
            )

        ax.set_xlim([-10, 10])
        ax.set_xticks([-2, -1, 1, 2])
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Response")
        ax.set_title(f"Block {block_id + 1}")
        ax.legend()

    fig.suptitle("Logistic Regression fits")
    plt.show()


def individual_logit_plots(data: pd.DataFrame) -> None:
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))

    x = np.linspace(-10, 10, 1000)
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    for i, ax in enumerate(axs.flatten()[:-1]):
        logit_params = []
        slot_machines = data["Slot Machine ID"].unique()
        slot_machines.sort()
        for sm in slot_machines:
            data_sm = data.loc[
                (data["Slot Machine ID"] == sm) & (data["participant_id"] == i + 1)
            ]
            reg = smf.logit(formula="Response ~ Difficulty", data=data_sm).fit(
                maxiter=1000
            )
            logit_params.append(list(reg.params))

        logit_params = np.asarray(logit_params)
        exp_b0 = np.exp(logit_params[:, 0])
        slopes = logit_params[:, 1] * exp_b0 / (1 + exp_b0) ** 2

        for j, (p, l) in enumerate(zip(logit_params, linestyles)):
            ax.plot(
                x,
                expit(p[1] * x + p[0]),
                label=f"SM {j}, slope {slopes[j]:.2f}",
                linestyle=l,
            )

        ax.set_xlim([-10, 10])
        ax.set_xticks([])
        ax.set_ylim([0, 1])
        ax.set_title(f"Participant {i+1}")
        if i > 15:
            ax.set_xlabel("Difficulty")
            ax.set_xticks([-2, -1, 1, 2])
        if i % 5 == 0:
            ax.set_ylabel("P(Response = Yes)")
        ax.legend()

    fig.suptitle("Non-hierarchical logistic Regression fits per subject")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/pilot_slot-machines_3/processed_data.csv")

    print("Running pooled logistic regression fits")
    pooled_logit_plots_per_block(df)
