import os
from dataclasses import dataclass

import dateutil.parser
import numpy as np
import pandas as pd


@dataclass
class CleanerRaw:
    """Clean the raw data folder to get rid of useless files."""

    path: str
    ref_date: str = "2000-01-01"
    min_size_bytes: int = 10_000

    def __post_init__(self):
        self.csv_files = [
            file for file in os.listdir(self.path) if file.endswith(".csv")
        ]

    def get_date(self, file: str) -> str:
        try:
            return file.split("_")[3]
        except dateutil.parser.ParserError:
            print(f"Warning: could not parse date from {file}; removed.")
            return self.ref_date

    @staticmethod
    def check_date_less_than(date: str, ref_date: str) -> list:
        date = dateutil.parser.parse(date)
        ref_date = dateutil.parser.parse(ref_date)
        return date < ref_date

    def is_not_csv(self, file: str) -> bool:
        return file not in self.csv_files

    def is_old(self, file: str) -> bool:
        return self.check_date_less_than(self.get_date(file), self.ref_date)

    def is_small(self, file: str) -> bool:
        return os.path.getsize(self.path + file) < self.min_size_bytes

    def clean(self) -> None:
        counter = 0
        for file in os.listdir(self.path):
            if self.is_not_csv(file) or self.is_old(file) or self.is_small(file):
                os.remove(self.path + file)
                print(f"Removed {file}.")
                counter += 1

        print(f"\nCleaned {self.path}:")
        print(f"Removed {counter} files. {len(os.listdir(self.path))} remain.\n")


@dataclass
class Columns:
    instructions = [
        "participant_id",
        "key_resp_instructions.keys",
        "key_resp_instructions.rt",
    ]
    data = [
        "participant_id",
        "state",
        "slot_machine_id",
        # "slot_machine_image",
        "slot_machine_mean_payoff",
        "price",
        "correct_response",
        "key_resp.keys",
        "key_resp.corr",
        "reward_drawn",
        "bonus_payout",
        "trials.thisN",
        "key_resp.rt",
    ]
    RENAMED_COLUMNS = {
        "key_resp.corr": "Choice Accuracy",
        "key_resp.rt": "Response Time",
        "slot_machine_id": "Slot Machine ID",
    }


class Processor:
    """Processes raw data into a clean format for analysis."""

    def __init__(self, path: str):
        self.path = path
        self.cols = Columns.data
        self.block_len = 192
        self.difficulty_map = {0: 2, 1: 1, 2: -1, 3: -2}
        self.response_map = {"left": 1, "right": 0}

    def _extract_data_partition(self, df: pd.DataFrame, filter_col=1) -> pd.DataFrame:
        """Extracts non-empty data given list of columns."""
        return df.loc[
            ~df.loc[:, self.cols[filter_col]].isnull(), self.cols
        ].reset_index(drop=True)

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns to be more readable."""
        return df.rename(columns=Columns.RENAMED_COLUMNS)

    def _add_block_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df["block_id"] = [int(i // (self.block_len)) + 1 for i in range(len(df))]
        df["block_type"] = ["training" if i == 1 else "test" for i in df["block_id"]]
        return df

    def _add_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Difficulty"] = df["state"] % 4
        df["Difficulty"].replace(self.difficulty_map, inplace=True)
        return df

    def _add_response(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Response"] = df["key_resp.keys"].replace(self.response_map)
        return df

    @staticmethod
    def _update_slot_machine_ids(df: pd.DataFrame) -> pd.DataFrame:
        df["Slot Machine ID"] = (df["Slot Machine ID"] + 1).astype(int)
        return df

    @staticmethod
    def _convert_object_cols_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
        string_cols = df.select_dtypes(include="object").columns
        df[string_cols] = df[string_cols].fillna("").astype("category")
        return df

    @staticmethod
    def _compute_expected_reward(df: pd.DataFrame) -> None:
        expected_reward_if_yes = df["slot_machine_mean_payoff"] - df["price"]
        df["expected_reward_if_correct"] = expected_reward_if_yes.clip(lower=0)
        df["expected_reward"] = expected_reward_if_yes * df["Response"]
        return df

    @staticmethod
    def _compute_accuracy(df: pd.DataFrame) -> pd.Series:
        return (df.groupby("participant_id")["Choice Accuracy"].mean() * 100).copy()

    @staticmethod
    def _compute_performance(df: pd.DataFrame) -> pd.Series:
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
        )

    @staticmethod
    def _shuffle_responses(df: pd.Series) -> pd.Series:
        """Shuffle the responses."""
        return df["Response"].sample(frac=1).reset_index(drop=True)

    @staticmethod
    def _generate_random_responses(df: pd.Series) -> pd.Series:
        """Responses generated by a random agent."""
        return np.random.choice([0, 1], size=len(df))

    def _get_null_distribution(self, df: pd.DataFrame, n: int = 100) -> pd.Series:
        """Compute the null distribution of performance scores."""
        print("Computing null distribution...")
        df = df.copy()
        performance_scores = []
        for _ in range(n):
            df["Response"] = self._generate_random_responses(df["Response"])
            df = self._compute_expected_reward(df)
            performance_scores.append(self._compute_performance(df))
        return pd.concat(performance_scores).sort_values()

    def _get_performance_threshold(
        self, df: pd.DataFrame, quantile_threshold=0.95
    ) -> pd.DataFrame:
        """Filter the data by performance score above quantile_threshold."""
        return self._get_null_distribution(df).quantile(quantile_threshold)

    def _compute_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for participants."""
        perf = self._compute_performance(df).to_frame(name="performance")
        perf["accuracy"] = self._compute_accuracy(df)
        perf["above_chance"] = perf["performance"] > self._get_performance_threshold(df)
        perf.sort_values("performance", ascending=False, inplace=True)
        return perf.reset_index()

    def _add_performance_metrics_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Computing and adding performance metrics to data...")
        perf = self._compute_performance_metrics(df)

        id_map = {k: v + 1 for k, v in zip(perf["participant_id"], perf.index)}
        perf_map = {k: v for k, v in zip(perf["participant_id"], perf["performance"])}
        acc_map = {k: v for k, v in zip(perf["participant_id"], perf["accuracy"])}
        ch_map = {k: v for k, v in zip(perf["participant_id"], perf["above_chance"])}

        df["id"] = df["participant_id"].map(id_map).astype(int)
        df["performance"] = df["participant_id"].map(perf_map).astype(float)
        df["accuracy"] = df["participant_id"].map(acc_map).astype(float)
        df["above_chance"] = df["participant_id"].map(ch_map).astype(bool)

        return df

    def _sort_by_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_performance_metrics_to_data(df)
        df.sort_values(by=["id", "block_id", "trials.thisN"], inplace=True)
        return df.reset_index(drop=True)

    def get_processed_data(self) -> pd.DataFrame:
        """Processes the raw data into a clean format for analysis."""
        files = os.listdir(self.path)
        dfs = []

        print("Reading and processing raw data files...")

        for file in files:
            df = pd.read_csv(f"{self.path}{file}")
            df = (
                df.pipe(self._extract_data_partition)
                .pipe(self._rename_columns)
                .pipe(self._add_block_id)
                .pipe(self._add_difficulty)
                .pipe(self._add_response)
                .pipe(self._update_slot_machine_ids)
                .pipe(self._compute_expected_reward)
            )

            dfs.append(df)

        df = pd.concat(dfs).reset_index(drop=True)

        return df.pipe(self._convert_object_cols_to_categorical).pipe(
            self._sort_by_performance
        )

    def extract_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns performance metrics for the processed data."""
        if "performance" not in df.columns:
            raise ValueError("Can only return performance metrics for processed data")

        perf = df.groupby("id")[["performance", "accuracy", "above_chance"]].mean()
        perf["above_chance"] = perf["above_chance"].astype(bool)

        return perf.reset_index()
