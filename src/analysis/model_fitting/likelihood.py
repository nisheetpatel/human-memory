from multiprocessing import Pool

import numpy as np
import pandas as pd
from pybads import BADS
from sklearn.model_selection import TimeSeriesSplit

from definitions import DATA_PATH
from src.data.transformer import DataTransformer
from src.simulation.models import DRA, Agent, EqualRA, FreqRA, StakesRA


class ResultsSaver:
    def __init__(self, data_transformer: DataTransformer):
        self.all_results_df = pd.DataFrame()
        self.data_transformer = data_transformer

    def store_all_results(self, all_results):
        print("Now storing results...")
        df = pd.DataFrame(
            [(result["id"], result["run_id"], result["agent_class"], result["fval"],
             *result["x"]) for result in all_results],
            columns=["Participant ID", "Run ID", "Model", "NLL", "lr_v", "lr_s", "lmda", "sigma_0"]
        )
        self.all_results_df = pd.concat([self.all_results_df, df]).reset_index(drop=True)

    def add_performance_scores_to_results(self) -> None:
        perf_unique = self.data_transformer.df.drop_duplicates(subset='participant_id')[
            ['participant_id', 'id', 'performance', 'accuracy', 'above_chance']
        ]
        self.all_results_df = pd.merge(self.all_results_df, perf_unique,
                 left_on='Participant ID', right_on='id', how='left')

    def save_results(self, save_path=DATA_PATH):
        # add performance scores to results
        self.add_performance_scores_to_results()

        # Save all results to a CSV file
        self.all_results_df.to_csv(save_path+"all_fit_results_fixed_sb_cross-validated.csv", index=False)

        best_model_counts = self.count_best_models(self.all_results_df)
        print(best_model_counts)

    def count_best_models(self, df: pd.DataFrame = None):
        if df is None:
            df = self.all_results_df

        # Group by 'Participant ID' and get the index of the min 'NLL' in each group
        best_model_index = df.groupby('Participant ID')['NLL'].idxmin()

        # Use this index to get the corresponding 'Model'
        best_models = df.loc[best_model_index, 'Model']

        # Count the number of each 'Model'
        counts = best_models.value_counts()

        return counts


class ModelFitter:
    # Constants
    N_RUNS = 3
    N_SPLITS = 5
    AGENT_CLASSES = [DRA, FreqRA, StakesRA, EqualRA]
    BOUNDS = {
        'lower_bounds': np.array([0.001, 0.001, 0.01, 1.]),
        'upper_bounds': np.array([0.5, 0.5, 1., 10.]),
        'plausible_lower_bounds': np.array([0.02, 0.02, 0.01, 2.]),
        'plausible_upper_bounds': np.array([0.2, 0.2, 1., 5.])
    }
    XI = [np.array([0.01, 0.025, 0.1, 2.5]), np.array([0.02, 0.05, 0.2, 2.5]),
          np.array([0.03, 0.075, 0.3, 2.5]), np.array([0.04, 0.05, 0.4, 2.5]),
          np.array([0.05, 0.025, 0.5, 2.5])]

    def compute_nll(self, x: np.ndarray, data: pd.DataFrame, agent_class: Agent) -> float:
        prob_actions_all = []
        agent = agent_class(x[0], x[1], x[2], x[3])

        for row in data.itertuples(index=False):
            # get agent's likelihood of choosing action given observations
            prob_actions = agent.action_prob(sm_id=int(row.sm_id), price=row.price)
            prob_actions_all.append(prob_actions[int(row.action)])

            # update agent's state
            agent.update(int(row.sm_id), row.price, row.reward, int(row.action))

        return -np.sum(np.log(np.clip(np.array(prob_actions_all), 0.001, 1)))

    def run_bads(self, target_fn, xi):
        bads = BADS(target_fn, xi, **self.BOUNDS)
        return bads.optimize()

    def generate_train_test_indices(self, df_participant):
        tscv = TimeSeriesSplit(n_splits=self.N_SPLITS)
        train_indices, test_indices = [], []
        for train_idx, test_idx in tscv.split(df_participant):
            train_indices.append(train_idx)
            test_indices.append(test_idx)
        return train_indices, test_indices
    
    def perform_cross_validation(self, df: pd.DataFrame, agent_class: Agent, run_id: int):
        train_indices, test_indices = self.generate_train_test_indices(df)
        test_nll_results = []

        for train_idx, test_idx in zip(train_indices, test_indices):
            # Define a target function to optimize using the training data for this fold
            def target_fn_for_this_fold(x):
                return self.compute_nll(x, df.iloc[train_idx], agent_class)
            
            # Optimize the parameters on the training set
            result = self.run_bads(target_fn_for_this_fold, self.XI[run_id])

            # Compute the NLL using the optimized parameters on the test data
            test_nll = self.compute_nll(result["x"], df.iloc[test_idx], agent_class)
            test_nll_results.append(test_nll)

        # Return avg NLL over test sets and the parameters that gave the lowest avg NLL
        return {"fval": np.mean(test_nll_results), "x": result["x"], "agent_class": agent_class.__name__}

    def fit_single_participant(self, df: pd.DataFrame):
        try:
            participant_all_results = []

            for agent_class in self.AGENT_CLASSES:

                # 5 runs for optimization
                for i in range(self.N_RUNS):
                    result = self.perform_cross_validation(df, agent_class, i)
                    result["id"] = df["id"].unique()[0]
                    result["run_id"] = i
                    participant_all_results.append(result)

            return participant_all_results
        except Exception as exception:
            print(f"An error occurred while fitting participant: {exception}")
            return [{"fval": np.nan, "x": (0,0,0,0), "id": df["id"].unique()[0], "run_id": i, "agent_class": agent_class.__name__}]

    def fit(self, results_saver: ResultsSaver, data: list[pd.DataFrame]):
        with Pool() as pool:
            for all_results in pool.imap_unordered(self.fit_single_participant, data):
                results_saver.store_all_results(all_results)


if __name__ == "__main__":
    data_transformer = DataTransformer()

    # Extract the participant data here
    ids = data_transformer.df["participant_id"].unique()
    data = [data_transformer.get_participant_data(id) for id in ids]

    # initialize the results saver
    results_saver = ResultsSaver(data_transformer=data_transformer)

    # initialize the fitter and fit data
    fitter = ModelFitter()
    fitter.fit(results_saver, data)
    results_saver.save_results()