"""Fit and classify all above-chance subjects with each supported optimizer and
print the resulting model-classification table.

    python -m src.scripts.run_model_fitting [--runs N] [--procs P] [--seed S]
"""

import argparse

import pandas as pd

from src.analysis.model_fitting.likelihood import (
    MODELS,
    classify,
    fit,
    load_participants,
)
from src.definitions import DATA_PATH

OPTIMIZERS = ["de", "lbfgsb", "powell", "cma", "bads"]
LABELS = {
    "de": "DE",
    "lbfgsb": "L-BFGS-B",
    "powell": "Powell",
    "cma": "CMA-ES",
    "bads": "BADS",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        type=int,
        default=0,
        help="restarts per optimizer (0 = each optimizer's default)",
    )
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    args = ap.parse_args()

    participants = load_participants(seed=args.seed)
    print(f"Fitting {len(participants)} subjects.\n")

    header = (
        f"{'optimizer':10s} "
        + " ".join(f"{MODELS[i]:>8s}" for i in MODELS)
        + f" {'DRA%':>6s} {'DRA/Freq':>9s}"
    )
    print(header)
    print("-" * len(header))

    summary = []
    for opt in args.optimizers.split(","):
        results = fit(
            opt,
            n_runs=(args.runs or None),
            procs=(args.procs or None),
            participants=participants,
        )
        results.to_csv(f"{DATA_PATH}model_fit_{opt}.csv", index=False)
        counts = classify(results).reindex(list(MODELS.values())).fillna(0).astype(int)
        n = int(counts.sum())
        dra, freq = counts["DRA"], counts["FreqRA"]
        print(
            f"{LABELS[opt]:10s} "
            + " ".join(f"{counts[MODELS[i]]:8d}" for i in MODELS)
            + f" {dra / n * 100:5.1f}% {dra / max(freq, 1):8.2f}x"
        )
        summary.append(
            {
                "optimizer": LABELS[opt],
                **counts.to_dict(),
                "DRA_pct": round(dra / n * 100, 1),
                "DRA_over_Freq": round(dra / max(freq, 1), 2),
            }
        )

    pd.DataFrame(summary).to_csv(f"{DATA_PATH}model_fit_summary.csv", index=False)
    print(f"\nSaved per-optimizer fits and summary to {DATA_PATH}")


if __name__ == "__main__":
    main()
