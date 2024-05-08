import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from scipy.special import expit


def plot_logit_preds(data, save_name: str = "model-preds.png"):
    sm_ids = data["sm_id"].unique()
    sm_ids.sort()
    models = data["Model"].unique()
    _, axs = plt.subplots(1, len(models), sharey=True, figsize=(len(models)*4,5))

    x = np.linspace(-10, 10, 1000)
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    axs[0].set_ylabel("Action")

    for model, ax in zip(models, axs):
        logit_params = []

        for sm in sm_ids:
            data_sm = data.loc[(data["sm_id"] == sm) & (data["Model"]==model)]
            reg = smf.logit(formula="action ~ price", data=data_sm).fit(maxiter=1000)
            logit_params.append(list(reg.params))

        logit_params = np.asarray(logit_params)
        exp_b0 = np.exp(logit_params[:, 0])
        slopes = logit_params[:, 1] * exp_b0 / (1 + exp_b0) ** 2

        for i, (p, ls) in enumerate(zip(logit_params, linestyles)):
            ax.plot(
                x,
                expit(p[1] * x + p[0]),
                label=f"$\\beta_{i+1}=${slopes[i]:.2f}",
                linestyle=ls,
            )

        ax.set_xlim([-5, 5])
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_xlabel("Price")
        ax.set_title(model)

        ax.legend()

    plt.savefig(save_name)
    plt.close()