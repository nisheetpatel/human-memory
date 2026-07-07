"""Likelihood-based fitting and selection for the resource-allocation models.

Each subject's four candidate models (DRA, FreqRA, StakesRA, EqualRA) are fit by
maximum likelihood on the test-phase choices, using 5-fold time-series
cross-validation with a warm start (the model learns over the training portion
and is scored on the held-out portion). A subject is classified as the model with
the best cross-validated negative log-likelihood.

Several black-box optimizers are supported so the classification can be compared
across fitting methods: BADS, differential evolution, L-BFGS-B, Powell and CMA-ES.
"""

import os
from multiprocessing import Pool

import cma
import numpy as np
import pandas as pd
from pybads import BADS
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import TimeSeriesSplit

from src.analysis.model_fitting import kernels
from src.data.processor import DataProcessor
from src.definitions import DATA_PATH

MODELS = {0: "DRA", 1: "FreqRA", 2: "StakesRA", 3: "EqualRA"}

BOUNDS = {
    "lower_bounds": np.array([0.001, 0.001, 0.01, 1.0]),
    "upper_bounds": np.array([0.5, 0.5, 1.0, 10.0]),
    "plausible_lower_bounds": np.array([0.02, 0.02, 0.01, 2.0]),
    "plausible_upper_bounds": np.array([0.2, 0.2, 1.0, 5.0]),
}
LB, UB = BOUNDS["lower_bounds"], BOUNDS["upper_bounds"]
PLB, PUB = BOUNDS["plausible_lower_bounds"], BOUNDS["plausible_upper_bounds"]
SCIPY_BOUNDS = list(zip(LB, UB))
X0 = [
    np.array([0.01, 0.025, 0.1, 2.5]),
    np.array([0.02, 0.05, 0.2, 2.5]),
    np.array([0.03, 0.075, 0.3, 2.5]),
    np.array([0.04, 0.05, 0.4, 2.5]),
    np.array([0.05, 0.025, 0.5, 2.5]),
]

# Per-run function-evaluation budget for BADS. Kept modest so the fit time stays
# reasonable across the whole panel of optimizers.
BADS_MAX_FUN_EVALS = 100

COLUMNS = [
    "Slot Machine ID",
    "slot_machine_mean_payoff",
    "price",
    "reward_drawn",
    "key_resp.keys",
    "id",
]


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """Map raw rows to the (sm_id, price, reward, action) fields the models use."""
    data = data.loc[:, COLUMNS].copy()
    action = data["key_resp.keys"].map({"right": 1, "left": 0})  # blanks/other -> NaN
    data = data.loc[action.notna()].copy()
    data["sm_id"] = data["Slot Machine ID"] - 1
    data["price"] = data["price"] - data["slot_machine_mean_payoff"]
    data["reward"] = data["reward_drawn"]
    data["action"] = action.loc[action.notna()].astype(int)
    return data.loc[:, ["sm_id", "price", "reward", "action"]].reset_index(drop=True)


def _arrays(dd: pd.DataFrame):
    return (
        dd["sm_id"].to_numpy(np.int64),
        dd["price"].to_numpy(np.float64),
        dd["reward"].to_numpy(np.float64),
        dd["action"].to_numpy(np.int64),
    )


def optimize(target, x0, method, run=0):
    """Minimise `target` from start `x0` with the requested optimizer."""
    if method == "bads":
        opts = {
            "display": "off",
            "uncertainty_handling": False,
            "max_fun_evals": BADS_MAX_FUN_EVALS,
        }
        result = BADS(target, np.array(x0, float), **BOUNDS, options=opts).optimize()
        return np.array(result["x"], float)
    if method == "de":
        return differential_evolution(
            target,
            SCIPY_BOUNDS,
            seed=0,
            maxiter=100,
            tol=1e-4,
            polish=True,
            init="sobol",
        ).x
    if method == "lbfgsb":
        return minimize(
            target,
            x0,
            method="L-BFGS-B",
            bounds=SCIPY_BOUNDS,
            options={"maxiter": 500, "eps": 1e-5},
        ).x
    if method == "powell":
        return minimize(
            target, x0, method="Powell", bounds=SCIPY_BOUNDS, options={"maxiter": 2000}
        ).x
    if method == "cma":
        es = cma.CMAEvolutionStrategy(
            np.array(x0, float),
            1.0,
            {
                "bounds": [list(LB), list(UB)],
                "CMA_stds": list(0.3 * (UB - LB)),
                "maxfevals": 4000,
                "tolfun": 1e-7,
                "verbose": -9,
                "seed": int(run) + 1,
            },
        )
        es.optimize(target)
        return np.array(es.result.xbest, float)
    raise ValueError(method)


class ModelFitter:
    N_SPLITS = 5
    # BADS' search is stochastic, so it is multi-started with different initial
    # conditions and the best fit is kept; the other optimizers are deterministic
    # or population-based and are run once.
    RESTARTS = {"bads": 5}

    def __init__(self, optimizer="bads", n_runs=None, n_splits=None):
        self.optimizer = optimizer
        self.n_runs = n_runs if n_runs is not None else self.RESTARTS.get(optimizer, 1)
        self.n_splits = n_splits or self.N_SPLITS

    def cross_validated_nll(self, mid, arrays, splits):
        """Best (over restarts) mean held-out NLL for one model."""
        sm, price, reward, action = arrays
        best_nll, best_x = np.inf, None
        for run in range(self.n_runs):
            fold_nlls, last_x = [], None
            for train_idx, test_idx in splits:
                train_obj = kernels.make_nll(
                    mid,
                    sm[train_idx],
                    price[train_idx],
                    reward[train_idx],
                    action[train_idx],
                )
                last_x = optimize(train_obj, X0[run], self.optimizer, run=run)
                end, n0 = int(test_idx[-1]) + 1, int(test_idx[0])
                test_obj = kernels.make_nll(
                    mid, sm[:end], price[:end], reward[:end], action[:end], n0=n0
                )
                fold_nlls.append(test_obj(last_x))
            mean_nll = float(np.mean(fold_nlls))
            if mean_nll < best_nll:
                best_nll, best_x = mean_nll, last_x
        return best_nll, best_x

    def fit_participant(self, task):
        pid, dd = task
        arrays = _arrays(dd)
        splits = list(
            TimeSeriesSplit(n_splits=self.n_splits).split(np.arange(len(arrays[0])))
        )
        rows = []
        for mid, name in MODELS.items():
            nll, x = self.cross_validated_nll(mid, arrays, splits)
            rows.append(
                {
                    "Participant ID": pid,
                    "Model": name,
                    "NLL": nll,
                    "lr_v": x[0],
                    "lr_s": x[1],
                    "lmda": x[2],
                    "sigma_0": x[3],
                }
            )
        return rows


def load_participants(seed=0, data_path=DATA_PATH):
    """Return [(participant_id, trials_df)] for above-chance subjects, test phase."""
    np.random.seed(seed)
    df = DataProcessor(path=data_path).get_processed_data()
    df = df.loc[df["above_chance"] & (df["block_type"] == "test")]
    return [
        (pid, transform(df.loc[df["participant_id"] == pid]))
        for pid in df["participant_id"].unique()
    ]


def _fit_worker(args):
    task, optimizer, n_runs, n_splits = args
    try:
        return ModelFitter(optimizer, n_runs, n_splits).fit_participant(task)
    except Exception as exc:  # keep the panel running if one subject fails
        return [
            {
                "Participant ID": task[0],
                "Model": "ERROR",
                "NLL": np.nan,
                "lr_v": np.nan,
                "lr_s": np.nan,
                "lmda": np.nan,
                "sigma_0": np.nan,
                "err": repr(exc),
            }
        ]


def fit(
    optimizer="bads", n_runs=None, n_splits=5, seed=0, procs=None, participants=None
):
    """Fit every subject with one optimizer and return the per-model results.
    `n_runs=None` uses each optimizer's default restart count (see ModelFitter)."""
    tasks = participants if participants is not None else load_participants(seed)
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    payload = [(t, optimizer, n_runs, n_splits) for t in tasks]
    rows = []
    with Pool(procs) as pool:
        for r in pool.imap_unordered(_fit_worker, payload):
            rows.extend(r)
    return pd.DataFrame(rows)


def classify(results: pd.DataFrame) -> pd.Series:
    """Best-cross-validated-NLL model per subject."""
    ok = results[results["Model"] != "ERROR"]
    best = ok.loc[ok.groupby("Participant ID")["NLL"].idxmin(), "Model"]
    return best.value_counts()
