from dataclasses import dataclass, field

import pandas as pd

from analysis.process import load_choice_data


def load_pilot_v3_data() -> pd.DataFrame:
    df_3 = load_choice_data("../data/pilot_slot-machines_3/")
    df_3a = load_choice_data("../data/pilot_slot-machines_3a/")
    df_3a["participant_id"] += 20
    return pd.concat([df_3, df_3a], ignore_index=True)


class DataProcessor:
    """Process the data and add performance scores it for analysis."""

    def __init__(self, df: pd.DataFrame = None) -> None:
        if not df:
            df = load_pilot_v3_data()

        self.df = self.compute_expected_reward(df)
        self.performance_metrics = self.compute_performance_metrics()

    @staticmethod
    def compute_expected_reward(df: pd.DataFrame) -> None:
        expected_reward_if_yes = df["slot_machine_mean_payoff"] - df["price"]
        df["expected_reward_if_correct"] = expected_reward_if_yes.clip(lower=0)
        df["expected_reward"] = expected_reward_if_yes * df["Response"]
        return df

    @staticmethod
    def compute_accuracy(df: pd.DataFrame) -> pd.Series:
        return (
            df[df["block_type"] == "test"]
            .groupby("participant_id")["Choice Accuracy"]
            .mean()
            * 100
        ).copy()

    @staticmethod
    def compute_performance(df: pd.DataFrame) -> pd.Series:
        max_expected_reward = (
            df.groupby(["participant_id"])["expected_reward_if_correct"]
            .mean()
            .unique()[0]
        )

        return (
            df[df["block_type"] == "test"]
            .groupby("participant_id")["expected_reward"]
            .mean()
            / max_expected_reward
            * 100
        ).copy()

    def get_null_distribution(self, n_shuffles: int = 100) -> pd.Series:
        """Compute the null distribution of performance scores."""
        print("Computing null distribution by shuffling data...")
        df = self.df.copy()
        performance_scores = []
        for _ in range(n_shuffles):
            df["Response"] = df["Response"].sample(frac=1).reset_index(drop=True)
            df = self.compute_expected_reward(df)
            performance_scores.append(self.compute_performance(df))
        return pd.concat(performance_scores).sort_values()

    def get_performance_threshold(self) -> pd.DataFrame:
        """Filter the data by performance score."""
        return self.get_null_distribution().quantile(0.997)

    def compute_performance_metrics(self) -> pd.DataFrame:
        """Compute the performance metrics for participants."""
        df = self.compute_performance(self.df).to_frame(name="performance")
        df["accuracy"] = self.compute_accuracy(self.df)
        df["above_chance"] = df["performance"] > self.get_performance_threshold()
        df.sort_values("performance", ascending=False, inplace=True)
        return df.reset_index()


@dataclass
class DataFilterer:
    data_processor = DataProcessor()

    def get_performance_metrics(self) -> pd.DataFrame:
        """Compute performance metrics to be returned as a dataframe."""
        return self.data_processor.performance_metrics

    @property
    def good_subjects_ids(self) -> pd.Series:
        return self.get_performance_metrics().query("above_chance == True")[
            "participant_id"
        ]

    def get_good_subjects_data(self) -> pd.DataFrame:
        """Get data for subjects who perform above chance level."""
        df = self.data_processor.df
        return df.loc[df["participant_id"].isin(self.good_subjects_ids)].copy()

    def get_bad_subjects_data(self) -> pd.DataFrame:
        """Get data for subjects who perform at or below chance level."""
        df = self.data_processor.df
        return df.loc[~df["participant_id"].isin(self.good_subjects_ids)].copy()


def main() -> DataFilterer:
    """Compute performance metrics and return a DataFilterer object."""
    return DataFilterer()


if __name__ == "__main__":
    main()
