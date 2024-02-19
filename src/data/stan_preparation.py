import numpy as np
import pandas as pd


def prepare_for_logit(data: pd.DataFrame, test_only=True) -> pd.DataFrame:
    """Prepare data for Stan (previously hierarchical.clean_data)."""

    data = data.copy()
    data.dropna(inplace=True)
    data["Response"] = data["Response"].astype(int)

    for i in range(1, 5):
        data[f"X{i}"] = data["Difficulty"] * (data["Slot Machine ID"] == i)

    if test_only:
        data = data.loc[data["block_type"] == "test"]

    data = data.loc[:, ["id", "Response", "X1", "X2", "X3", "X4"]]

    return data


def get_choice_data_dict(data: pd.DataFrame) -> dict:
    """Get choice data dict to be fed into stan."""

    y = data["Response"].astype(int)
    X = data.loc[:, data.columns.isin(["X1", "X2", "X3", "X4"])]
    p_id = data["id"]

    return {
        "N": X.shape[0],  # number of training samples
        "K": X.shape[1],  # number of predictors
        "L": len(p_id.unique()),  # number of levels/subjects
        "y": y.values.tolist(),  # response variable
        "X": np.array(X),  # matrix of predictors
        "ll": np.array(p_id.values),  # subject id
        "ss": np.array(X != 0, dtype=int),  # slot machine id indicator
    }


def prepare_for_stan(data: pd.DataFrame, test_only=True):
    return get_choice_data_dict(prepare_for_logit(data, test_only))