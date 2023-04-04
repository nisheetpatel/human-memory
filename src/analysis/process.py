import os
from dataclasses import dataclass

import dateutil.parser
import pandas as pd


def get_file_date(file: str) -> str:
    return file.split("_")[3]


def check_if_date_greater_than(date: str, ref_date: str) -> list:
    date = dateutil.parser.parse(date)
    ref_date = dateutil.parser.parse(ref_date)
    return date > ref_date


def get_all_csv_files(path: str, min_size_bytes=10_000) -> list:
    files = os.listdir(path)
    csv_files = [
        file
        for file in files
        if (file.endswith(".csv") & (os.path.getsize(path + file) > min_size_bytes))
    ]
    print(f"\nRetrieved {len(csv_files)} csv files from {path}.\n")

    return csv_files


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
        "slot_machine_image",
        "slot_machine_mean_payoff",
        "price",
        "correct_response",
        "key_resp.keys",
        "key_resp.corr",
        "reward_drawn",
        # "reward_received",
        "bonus_payout",
        "trials.thisN",
        "key_resp.rt",
    ]


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Renames columns to be more readable."""
    return data.rename(
        columns={
            "key_resp.corr": "Choice Accuracy",
            "key_resp.rt": "Response Time",
            "slot_machine_id": "Slot Machine ID",
        },
    )


def extract_data_partition(df: pd.DataFrame, cols: list, filter_col=1) -> pd.DataFrame:
    """Extracts non-empty data given list of columns."""
    return df.loc[~df.loc[:, cols[filter_col]].isnull(), cols].reset_index(drop=True)


def add_block_id(df: pd.DataFrame, block_len: int = 192) -> pd.DataFrame:
    df["block_id"] = [int(i // (block_len)) for i in range(len(df))]
    df["block_type"] = ["training" if i == 0 else "test" for i in df["block_id"]]
    return df


def add_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    df["Difficulty"] = df["state"] % 4
    df["Difficulty"].replace({0: 2, 1: 1, 2: -1, 3: -2}, inplace=True)
    return df


def add_response(df: pd.DataFrame) -> pd.DataFrame:
    df["Response"] = df["key_resp.keys"].replace({"left": 1, "right": 0})
    return df


def check_length(df: pd.DataFrame) -> bool:
    if len(df) < 192 * 2:
        # raise ValueError("Dataframe is not of the correct length.")
        return False
    return True


def print_trial_splits(df: pd.DataFrame) -> None:
    print(
        df.groupby(["Slot Machine ID", "slot_machine_image"])[
            "Difficulty"
        ].value_counts()
        / 4
    )


def get_dataframes(csv_files: list, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a tuple of dataframes for data and instructions."""

    dfs_data = []
    dfs_instr = []
    files_skipped = 0

    for file in csv_files:
        df = pd.read_csv(path + file)

        df_instr = extract_data_partition(df, Columns.instructions)
        df_data = extract_data_partition(df, Columns.data)

        df_data = (
            df_data.pipe(add_block_id)
            .pipe(rename_columns)
            .pipe(add_difficulty)
            .pipe(add_response)
        )

        # print_trial_splits(df_data)
        if check_length(df_data):
            dfs_data.append(df_data)
            dfs_instr.append(df_instr)
        else:
            print(f"Skipping {file}.")
            files_skipped += 1

    df_data = pd.concat(dfs_data).reset_index(drop=True)
    df_instr = pd.concat(dfs_instr).reset_index(drop=True)
    print(f"\nSkipped {files_skipped} files.\n")

    return df_data, df_instr


def load_all_data(path: str) -> None:
    csv_files = get_all_csv_files(path=path)
    df_data, df_instr = get_dataframes(csv_files, path=path)

    participant_ids = df_data["participant_id"].unique()
    id_map = {id: i + 1 for i, id in enumerate(participant_ids)}

    for df in [df_data, df_instr]:
        df["participant_id"] = df["participant_id"].map(id_map)

    return df_data, df_instr


def load_choice_data(path: str) -> None:
    df_data, _ = load_all_data(path=path)
    return df_data


def load_instructions_data(path: str) -> None:
    _, df_instr = load_all_data(path=path)
    return df_instr


if __name__ == "__main__":
    print("Not meant to be run as a script. To load choice data, run:")
    print('load_choice_data(path="./data/pilot_slot-machines_3a/")\n')
    print("To load instructions data, run:")
    print('load_instructions_data(path="./data/pilot_slot-machines_3a/")\n')
    print("To load all data, run:")
    print('load_all_data(path="./data/pilot_slot-machines_3a/")\n')
