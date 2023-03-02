import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from scipy.special import expit  # pylint: disable=no-name-in-module

from ..customtypes import ModelName


def clean_data_for_stan(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(inplace=True)
    data["Slot Machine ID"] = (data["Slot Machine ID"] + 1).astype(int)
    data["Response"] = data["Response"].astype(int)
    data["participant_id"] += 1

    for i in range(1, 5):
        data[f"X{i}"] = data["Difficulty"] * (data["Slot Machine ID"] == i)

    data = data.loc[
        data["block_id"] > 0,
        ["participant_id", "Response", "X1", "X2", "X3", "X4"],
    ]
    return data


BAYESIAN_HIERARCHICAL_LOGIT_STAN = """
    data {
        int<lower=1> N; // number of training samples
        int<lower=0> K; // number of predictors (4 SMs)
        int<lower=1> L; // number of subjects
        
        int<lower=1, upper=L> ll[N]; // subject id {1,...,L}
        //row_vector<lower=0, upper=1>[K] ss[N]; // slot machine id indicator

        row_vector[K] X[N];         // predictors
        int<lower=0, upper=1> y[N]; // response
    }

    parameters {
        vector[K] beta[L];  // individual slope
        vector[K] mu_beta;  // Hierarchical mean for slope
        vector<lower=0>[K] sigma_beta; // h std for slope
        
        //vector[K] alpha[L]; // individual intercept
        //vector[K] mu_alpha;   // Hierarchical mean for intercept
        //vector<lower=0>[K] sigma_alpha; // h std for intercept

        vector[L] alpha; // common intercept
        real mu_alpha;   // hierarchical mean for common intercept
        real<lower=0> sigma_alpha; // h std for common intercept
    }

    model {
        mu_beta ~ normal(0.25,0.25);
        sigma_beta ~ cauchy(0,0.5);
            
        mu_alpha ~ normal(0, 0.25);
        sigma_alpha ~ cauchy(0, 0.5);
        
        //mu_alpha_0 ~ normal(0, 0.25);
        //sigma_alpha_0 ~ cauchy(0, 0.5);

        for (l in 1:L) {
            beta[l] ~ normal(mu_beta, sigma_beta);
            alpha[l] ~ normal(mu_alpha, sigma_alpha);
            //alpha_0[l] ~ normal(mu_alpha_0, sigma_alpha_0);
        }
        
        {
        vector[N] x_beta_ll;

        for (n in 1:N)
            x_beta_ll[n] = X[n] * beta[ll[n]] + alpha[ll[n]];
        // x_beta_ll[n] = X[n] * beta[ll[n]] + ss[n] * alpha[ll[n]] + alpha_0[ll[n]];
        
        y ~ bernoulli_logit(x_beta_ll);
        }
    }
"""


def fit_posterior(data, n_samples=25_000, n_chains=4):
    # extract dataframe (relevant subjects)
    posterior_samples = []

    y = data["Response"].astype(int)
    X = data.loc[:, data.columns.isin(["X1", "X2", "X3", "X4"])]
    p_id = data["participant_id"]

    # defining the choice data
    choice_data = {
        "N": X.shape[0],  # number of training samples
        "K": X.shape[1],  # number of predictors
        "L": len(p_id.unique()),  # number of levels/subjects
        "y": y.values.tolist(),  # response variable
        "X": np.array(X),  # matrix of predictors
        "ll": np.array(p_id.values),  # subject id
        # "ss": np.array(X != 0, dtype=int),  # slot machine id indicator
    }

    # fit, extract parameters, and print summary
    posterior = stan.build(BAYESIAN_HIERARCHICAL_LOGIT_STAN, data=choice_data)
    samples = posterior.sample(num_chains=n_chains, num_samples=n_samples)

    posterior_samples.append(samples)

    return posterior_samples


# Threshold analysis
def test(b, testFor: ModelName.DRA, e_factor=1, th_factor=0):
    # threshold: std. dev of posterior x th_factor
    stdev = np.median(np.std(b, axis=1))
    th = stdev * th_factor
    e = stdev * e_factor

    if testFor == ModelName.DRA:
        c1 = b[0] - np.max(b[1:], axis=0) > th
        c2 = np.min(b[:3], axis=0) - b[3] > th
        c = c1 * c2

    elif testFor == ModelName.FREQ:
        c1 = abs(b[0] - b[1]) < e
        c2 = abs(b[2] - b[3]) < e
        c3 = np.min(b[:2], axis=0) - np.max(b[2:], axis=0) > th
        c = c1 * c2 * c3

    elif testFor == ModelName.STAKES:
        c1 = abs(b[0] - b[2]) < e
        c2 = abs(b[1] - b[3]) < e
        c3 = np.min(b[[0, 2], :], axis=0) - np.max(b[[1, 3], :], axis=0) > th
        c = c1 * c2 * c3

    elif testFor == ModelName.EQUALPRECISION:
        c1 = abs(b[0] - b[1]) < e
        c2 = abs(b[0] - b[2]) < e
        c3 = abs(b[0] - b[3]) < e
        c4 = abs(b[1] - b[2]) < e
        c5 = abs(b[1] - b[3]) < e
        c6 = abs(b[2] - b[3]) < e
        c = c1 * c2 * c3 * c4 * c5 * c6

    else:
        raise ValueError("Invalid model type!")

    pass_percent = 100 * np.sum(c) / len(c)

    return pass_percent


def prob_of_signature(b, e_factor=0.5, th_factor=0):
    posterior_prob_matrix = []

    pass_percent = []

    for model in ModelName:
        pass_percent += [test(b, testFor=model, e_factor=e_factor, th_factor=th_factor)]

    posterior_prob_matrix += [np.array(pass_percent) / np.sum(pass_percent)]

    return np.around(posterior_prob_matrix, 2)


def plot_prob_model_signature(betas: np.ndarray):
    es = [0.25, 0.5, 1, 2]
    models = [m.value for m in ModelName]
    fig, axs = plt.subplots(1, len(es), figsize=(20, 5))

    for e, ax in zip(es, axs):
        prob_model = prob_of_signature(betas, e_factor=e)[0]
        ax.bar(models, prob_model)
        symbol_str = "$\sigma_\mu$"  # pylint: disable=anomalous-backslash-in-string
        ax.set_title(f"Similarity threshold = {e}{symbol_str}")
        ax.set_ylim([0, 1])

    axs[0].set_ylabel("Probability of model signature")
    fig.suptitle("Hierarchical logistic regression: model signatures at group level")

    plt.savefig("../../figures/slot-machines/pilot_3/prob_model_signature_group.png")
    plt.savefig("../../figures/slot-machines/pilot_3/prob_model_signature_group.svg")


def plot_prob_model_signature_individual(betas: np.ndarray):
    for e in [0.25, 0.5, 1, 2]:
        models = ["DRA", "Freq", "Stakes", "EP"]
        fig, axs = plt.subplots(4, 5, figsize=(20, 16))

        for i, (beta, ax) in enumerate(zip(betas, axs.flatten())):
            prob_model = prob_of_signature(beta, e)[0]
            ax.bar(models, prob_model)
            ax.set_title(f"Participant = {i}")
            ax.set_ylim([0, 1])
            if i % 5 == 0:
                ax.set_ylabel("Probability of model signature")
            if i < 15:
                ax.set_xticklabels("")
        fig.suptitle(
            f"Hierarchical logistic regression: model signatures per subject\nSimilarity threshold = {e}sigma_mu"
        )
        plt.savefig(
            f"../../figures/slot-machines/pilot_3/individual/prob_model_signature_0p{int(e*100)}.png"
        )
        plt.savefig(
            f"../../figures/slot-machines/pilot_3/individual/prob_model_signature_0p{int(e*100)}.svg"
        )
        plt.close()


def plot_logistic_curves_individual_wo_bias(betas: np.ndarray):
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    x = np.linspace(-10, 10, 1000)
    fig, axs = plt.subplots(4, 5, figsize=(20, 15))
    for i, (beta, ax) in enumerate(zip(betas, axs.flatten())):
        for sm in range(4):
            ax.plot(x, expit(beta[sm] * x), label=sm, linestyle=linestyles[sm])
        ax.set_xlim([-10, 10])
        ax.set_ylim([0, 1])
        ax.set_title(f"Participant = {i}")
        if i < 15:
            ax.set_xticklabels("")
        if i % 5 == 0:
            ax.set_ylabel("P(Response = Yes)")
    fig.legend(["SM 1", "SM 2", "SM 3", "SM 4"], loc="center right")
    fig.suptitle("Hierarchical logistic regression fits (bias corrected)")
    plt.savefig(
        "../../figures/slot-machines/pilot_3/individual/logit_fits_bias-corrected.png"
    )
    plt.savefig(
        "../../figures/slot-machines/pilot_3/individual/logit_fits_bias-corrected.svg"
    )
    plt.close()


def plot_logistic_curves_individual(alphas: np.ndarray, betas: np.ndarray):
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    x = np.linspace(-10, 10, 1000)
    fig, axs = plt.subplots(4, 5, figsize=(20, 15))
    for i, (alpha, beta, ax) in enumerate(zip(alphas, betas, axs.flatten())):
        for sm in range(4):
            ax.plot(
                x, expit(alpha[sm] + beta[sm] * x), label=sm, linestyle=linestyles[sm]
            )
        ax.set_xlim([-10, 10])
        ax.set_ylim([0, 1])
        ax.set_title(f"Participant = {i}")
        if i < 15:
            ax.set_xticklabels("")
        if i % 5 == 0:
            ax.set_ylabel("P(Response = Yes)")
    fig.legend(["SM 1", "SM 2", "SM 3", "SM 4"], loc="center right")
    fig.suptitle("Hierarchical logistic regression fits")
    plt.savefig("../../figures/slot-machines/pilot_3/individual/logit_fits.png")
    plt.savefig("../../figures/slot-machines/pilot_3/individual/logit_fits.svg")
    plt.close()


def density_plots(samples):
    import seaborn as sns

    beta_indivs = samples["beta"]
    slot_machine_id = np.tile(np.repeat(np.arange(4), 100_000), 19)
    participant_ids = np.repeat(np.arange(19), 4 * 100_000)
    df_samples = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "slot_machine_id": slot_machine_id,
            "beta": beta_indivs.flatten(),
        }
    )

    _, axs = plt.subplots(4, 5, figsize=(20, 16))
    for i, ax in enumerate(axs.flatten()[:-1]):
        sns.kdeplot(
            data=df_samples.loc[df_samples["participant_id"] == i],
            x="beta",
            hue="slot_machine_id",
            palette="tab10",
            ax=ax,
        )
        if i < 15:
            ax.set_xlabel("")
        if i % 5 != 0:
            ax.set_ylabel("")
        ax.set_title(f"Participant = {i}")
        ax.get_legend().remove()
    plt.show()


def main(data_path: str):
    from .process import load_choice_data

    df = load_choice_data(path=data_path)
    df = clean_data_for_stan(df)

    # fit model
    samples = fit_posterior(df)

    # plot at group level
    betas = np.asarray([samples[i]["mu_b"][0] for i in range(4)])
    plot_prob_model_signature(betas)

    # plot at individual level
    beta_indiv = np.asarray([samples[i]["beta"] for i in range(4)])
    s = beta_indiv.shape
    beta_indiv = beta_indiv.reshape((s[0], s[1], s[-1])).transpose(1, 0, 2)
    plot_prob_model_signature_individual(beta_indiv)
    plot_logistic_curves_individual_wo_bias(np.mean(beta_indiv, axis=-1))

    alpha_indiv = np.asarray([samples[i]["alpha"] for i in range(4)])
    s = alpha_indiv.shape
    alpha_indiv = alpha_indiv.reshape((s[0], s[1], s[-1])).transpose(1, 0, 2)
    plot_logistic_curves_individual(
        np.mean(alpha_indiv, axis=-1), np.mean(beta_indiv, axis=-1)
    )


if __name__ == "__main__":
    import os

    if os.path.exists("../../data/"):
        main("../../data/pilot_slot-machines_3a/")
    else:
        print("No data found. Are you in the root directory?")
