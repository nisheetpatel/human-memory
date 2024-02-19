import numpy as np
import pandas as pd

from .processor import get_processed_data


class DataTransformer:
    COLUMNS_OF_INTEREST = [
        "Slot Machine ID",
        "slot_machine_mean_payoff",
        "price",
        "reward_drawn",
        "key_resp.keys",
        "id"
    ]

    def __init__(self, get_data: callable = get_processed_data):
        df = get_data()
        self.df = df.loc[df["above_chance"]]

    @staticmethod
    def transform_data(data: pd.DataFrame) -> pd.DataFrame:
        data['key_resp.keys'].replace('', np.nan, inplace=True)
        data.dropna(subset=['key_resp.keys'], inplace=True)
        data["sm_id"] = data["Slot Machine ID"] - 1
        data["price"] = data["price"] - data["slot_machine_mean_payoff"]
        data["reward"] = data["reward_drawn"]
        data["action"] = data["key_resp.keys"].map({"right": 1, "left": 0}).astype(int)
        return data.loc[:, ["sm_id", "price", "reward", "action", "id"]]

    def get_participant_data(self, participant_id) -> pd.DataFrame:
        return self.transform_data(self.df.loc[
            self.df["participant_id"] == participant_id, 
            self.COLUMNS_OF_INTEREST]
        )