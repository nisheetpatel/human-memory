import os
import numpy as np
import pandas as pd


def extract_data():
    dfs, dfs_pmt = [], []

    for subject_id in range(1, 6):
        # read csv data files
        directory = f"./pilot data/{subject_id}/"

        # read all csv files and put them in the list
        df_list = []

        for root, dirs, files in os.walk(directory):

            for file in files:

                if file.endswith(".csv"):
                    df_list += [pd.read_csv(directory + file)]

                elif file.endswith(".npy"):
                    delta_subject = np.around(np.load(directory + file), 1)

        # merge all sessions and define state
        df = pd.concat(df_list)
        df["delta"] = delta_subject
        df["State"] = ((df["Condition"] % 12) // 3) + 1
        df["Choice accuracy"] = df["correct"].astype(float)

        # extract pmt trials only
        df_pmt = df.loc[
            (df["bonus_trial"] == True) & (df["Session ID"] > 0)
        ].reset_index()

        # save
        # save_path = f"./pilot data/subject_{subject_id}_pmt.csv"
        # df_pmt.to_csv(save_path)

        # adding both to the list of all subjects
        dfs += [df]
        dfs_pmt += [df_pmt]

    # concatenate all subject dfs
    df = pd.concat(dfs).reset_index()
    df_pmt = pd.concat(dfs_pmt).reset_index()
    df_test = df.loc[df["Session type"] == "testing"].reset_index()

    return df, df_pmt, dfs, dfs_pmt


if __name__ == "__main__":
    df, df_pmt, _ = extract_data()

    from .plotting import plot_diagnostics

    plot_diagnostics(df=df, df_pmt=df_pmt)
