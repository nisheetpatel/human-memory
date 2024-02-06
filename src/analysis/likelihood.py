import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from pybads import BADS

from analysis.data import Processor as DataProcessor
from definitions import DATA_PATH
from model.model_new import DDRA, Agent, DEqualRA, DFreqRA, DStakesRA

# read and process data, then extract performance metrics
data_processor = DataProcessor(path=DATA_PATH+"pilot_slot-machines_3/")
df = data_processor.get_processed_data()
df_good = df.loc[df["above_chance"]]
perf = data_processor.extract_performance_metrics(df)


# extract data for likelihood calculation
# format required: (sm_id, price, reward, action) x N for each subject
# potentially also required: slot_machine_mean_payoff

columns_of_interest = [
    "Slot Machine ID",
    "slot_machine_mean_payoff",
    "price",
    "reward_drawn",
    "key_resp.keys"
]

def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    data['key_resp.keys'].replace('', np.nan, inplace=True)
    data.dropna(subset=['key_resp.keys'], inplace=True)
    data["sm_id"] = data["Slot Machine ID"] - 1
    data["price"] = data["price"] - data["slot_machine_mean_payoff"]
    data["reward"] = data["reward_drawn"]
    data["action"] = data["key_resp.keys"].map({"right": 1, "left": 0}).astype(int)
    return data.loc[:, ["sm_id", "price", "reward", "action"]]

# test for one participant
participant = df.loc[0, "participant_id"]
df_0 = transform_data(
    df.loc[df["participant_id"]==participant, columns_of_interest]
)


# likelihood function adapted for BADS / VBMC
def compute_NLL(x: np.ndarray, data: pd.DataFrame, agent_class: Agent) -> float:
    log_likelihood = 0.0
    agent = agent_class(lr=x[0], lmda=x[1], sigma_base=x[2], sigma_0=x[3])

    for _, row in data.iterrows():
        # get agent's likelihood of choosing action given observations
        prob_actions = agent.action_prob(sm_id=int(row["sm_id"]), price=row["price"])
        log_likelihood += np.log(max(prob_actions[int(row["action"])],0.001))

        # update agent stuff
        agent.update(int(row["sm_id"]), row["price"], row["reward"], int(row["action"]))

    return -log_likelihood

lower_bounds = np.array([0.001, 0.001, 3., 1.])
upper_bounds = np.array([0.999, 10., 50., 5.])
plausible_lower_bounds = np.array([0.01, 0.025, 5., 1.])
plausible_upper_bounds = np.array([0.4, 1, 10., 5.])
x0 = np.array([0.05, 0.1, 5, 2])

agent_classes = [DDRA, DFreqRA, DStakesRA, DEqualRA]
# results = []

# for agent_class in agent_classes:
#     target_fn = partial(compute_NLL, data=df_0, agent_class=agent_class)
#     bads = BADS(target_fn, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds)
#     optimize_result = bads.optimize()
#     results.append(optimize_result)

# for i, result in enumerate(results):
#     print(f"{agent_classes[i].__name__} NLL {result['fval']}, x {result['x']}")


################################################################

def run_bads_multiple_times(target_fn, x0, n_runs=5):
    results = []
    for _ in range(n_runs):
        bads = BADS(target_fn, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds)
        optimize_result = bads.optimize()
        results.append(optimize_result)
    return results


def run_optimization_for_participant(participant_id, df_participant, agent_classes):
    participant_results = []

    for agent_class in agent_classes:
        target_fn = partial(compute_NLL, data=df_participant, agent_class=agent_class)
        agent_results = run_bads_multiple_times(target_fn, x0, n_runs=5)
        best_result = min(agent_results, key=lambda result: result["fval"])
        participant_results.append(best_result)

    return participant_id, participant_results


# Create a pool of worker processes
pool = mp.Pool()

# Loop over each participant
participant_ids = df["participant_id"].unique()
results_by_participant = pool.starmap(
    run_optimization_for_participant,
    [(participant_id, df_participant, agent_classes) for participant_id, df_participant in df.groupby("participant_id")]
)

# Close the pool to prevent any more tasks from being submitted to the pool
pool.close()



# Save the results to a CSV file
results_df = pd.DataFrame(
    [(participant_id, agent_classes[j].__name__, result["fval"], result["x"]) for participant_id, participant_results in results_by_participant for j, result in enumerate(participant_results)],
    columns=["Participant ID", "Model", "NLL", "Parameters"]
)
results_df.to_csv("optimization_results.csv", index=False)