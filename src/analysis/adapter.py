import numpy as np
import pandas as pd

from analysis.data import Processor as DataProcessor
from definitions import DATA_PATH
from model.model_new import DDRA, Agent

# read and process data, then extract performance metrics
data_processor = DataProcessor(path=DATA_PATH)
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

# compute log likelihood of data for model

# initialize log likelihood
log_likelihood = 0.0
agent: Agent = DDRA()

for i, data in df_0.iterrows():
    # get agent's likelihood of choosing action given observations
    prob_actions = agent.action_prob(sm_id=int(data["sm_id"]), price=data["price"])
    log_likelihood += np.log(prob_actions[int(data["action"])])

    # update agent stuff
    agent.update(int(data["sm_id"]), data["price"], data["reward"], int(data["action"]))

