import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

ComposableFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


@dataclass
class ProcessingError(Exception):
    error: str


@dataclass
class DataProcessor:
    subject_id: int

    def extract(self, directory=Optional[str]) -> pd.DataFrame:
        if not directory in locals():
            directory = f"./data/pilot_data/{self.subject_id}/"

        # read all csv files and put them in the list
        df_list = []

        for _, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    df_list += [pd.read_csv(directory + file)]
                elif file.endswith(".npy"):
                    delta_subject = np.around(np.load(directory + file), 1)

        # merge all sessions
        df = pd.concat(df_list)
        df["delta"] = delta_subject
        self.df = df

        return df

    def check_data(self, error_msg: str = "use") -> None:
        if not hasattr(self, "df"):
            raise ProcessingError(f"Cannot {error_msg} data before extracting!")

    def slice(self, rows, error_msg: str = "slice") -> pd.DataFrame:
        self.check_data(error_msg=error_msg)
        return self.df.loc[rows].reset_index()

    @staticmethod
    def _drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.drop(
            columns=[
                ".thisRepN",
                ".thisTrialN",
                ".thisIndex",
                "DateTime",
                "Unnamed: 17",
            ],
            inplace=True,
        )
        return df

    @staticmethod
    def _add_state_column(df: pd.DataFrame) -> pd.DataFrame:
        df["State"] = ((df["Condition"] % 12) // 3) + 1
        return df

    @staticmethod
    def _add_choice_accuracy_column(df: pd.DataFrame) -> pd.DataFrame:
        df["Choice accuracy"] = df["correct"].astype(float)
        return df

    @staticmethod
    def _define_session_order(df: pd.DataFrame) -> pd.DataFrame:
        df["Session type"] = pd.Categorical(
            df["Session type"],
            categories=["practice", "training", "testing"],
            ordered=True,
        )
        return df

    @staticmethod
    def _sort_by_session(df: pd.DataFrame) -> pd.DataFrame:
        df.sort_values(
            by=["Day", "Session type", "Session ID"], inplace=True, ignore_index=True
        )
        return df

    @staticmethod
    def _add_new_session_column(df: pd.DataFrame) -> pd.DataFrame:
        df["new_session"] = df[".thisN"].diff() < 0
        return df

    @staticmethod
    def _add_new_day_column(df: pd.DataFrame) -> pd.DataFrame:
        df["new_day"] = df["Day"].diff() == 1
        return df

    @staticmethod
    def _add_difficulty_column(df: pd.DataFrame) -> pd.DataFrame:
        dict_map = {False: "easy 2AFC", True: "difficult 2AFC"}
        df["difficulty"] = (df["Condition"] % 3) != 1
        df["difficulty"].replace(dict_map, inplace=True)
        df.loc[df["bonus_trial"] == True, "difficulty"] = "bonus"
        return df

    @staticmethod
    def _define_difficulty_order(df: pd.DataFrame) -> pd.DataFrame:
        df["difficulty"] = pd.Categorical(
            df["difficulty"],
            categories=["easy 2AFC", "difficult 2AFC", "bonus"],
            ordered=True,
        )
        return df

    def preprocess(self) -> pd.DataFrame:
        self.check_data(error_msg="process")
        preprocess_data = compose(
            self._drop_useless_columns,
            self._add_state_column,
            self._add_choice_accuracy_column,
            self._add_difficulty_column,
            self._define_difficulty_order,
            self._define_session_order,
            self._sort_by_session,
            self._add_new_session_column,
            self._add_new_day_column,
        )
        self.df = preprocess_data(self.df)
        return self.df

    def define_pmt_data(self) -> pd.DataFrame:
        rows = (self.df["bonus_trial"] == True) & (self.df["Session ID"] > 0)
        self.df_pmt = self.slice(rows, error_msg="define PMT")
        return self.df_pmt

    def define_test_data(self) -> pd.DataFrame:
        rows = (self.df["bonus_trial"] == False) & (
            self.df["Session type"] == "testing"
        )
        self.df_test = self.slice(rows, error_msg="define test")
        return self.df_test

    def compute_moving_average(self) -> None:
        pass
