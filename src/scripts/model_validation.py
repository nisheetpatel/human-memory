import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.processor import get_processed_data
from src.definitions import DATA_PATH
from src.simulation.experiment import ExperimentBestFitParams, RAModelParams
from src.simulation.models import DRA, Agent, EqualRA, FreqRA, StakesRA

# get the best-fitting parameters for each subject
########################################################################

# load the parameters
df_fitted_params = pd.read_csv(DATA_PATH + "all_fit_results_fixed_sb_cross-validated_new.csv")

# function to extract best-fitting parameters for each subject/model
def get_best_params(participant_id: int, model: Agent, df: pd.DataFrame = df_fitted_params) -> RAModelParams:
    """
    Extract the best fitting parameters for a given participant and model.

    Args:
        df (pd.DataFrame): The DataFrame containing the fitted parameters.
        participant_id (int): The ID of the participant.
        model: Model of type Agent.

    Returns:
        RAModelParams: An instance of RAModelParams with the best fitting parameters.
    """
    # Filter the DataFrame for the specific participant and model
    participant_data = df[(df['Participant ID'] == participant_id) & (df['Model'] == model.__name__)]

    if participant_data.empty:
        raise ValueError(f"No data found for Participant ID {participant_id} and Model {model.__name__}")

    # Find the row with the lowest NLL
    best_fit = participant_data.loc[participant_data['NLL'].idxmin()]

    # Create and return an instance of RAModelParams
    return RAModelParams(
        lmda=best_fit['lmda'],
        sigma_base=5.0,  # As specified, this is a constant
        lr_s=best_fit['lr_s'],
        lr_v=best_fit['lr_v']
    )

########################################################################
# get the trial data
########################################################################

# load data
df = get_processed_data(data_path=DATA_PATH + "pilot_slot-machines_3/")

# define function to extract pre-defined trial parameters
def extract_trial_data(participant_id: int) -> list[tuple[int, float, float]]:
    """
    Extract trial data for a specific participant from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the trial data.
        participant_id (int): The ID of the participant to extract data for.

    Returns:
        list[tuple[int, float, float]]: A list of tuples containing (state, rtrn, price) for each trial.
    """
    # Filter the DataFrame for the specific participant
    participant_data = df[df['id'] == participant_id]

    # Extract the required columns and create the list of tuples
    trial_data = [
        (
            int(row['state']),  # Convert state to int
            row['price'] - row['slot_machine_mean_payoff'],
            row['reward_drawn']
        )
        for _, row in participant_data.iterrows()
    ]

    return trial_data


########################################################################
# define and simulate models
########################################################################

def get_params_and_data_for_model(model: str) -> dict[int, dict]:
    return {
        i: {
            "params": get_best_params(i, model),
            "data": extract_trial_data(i)
        }
        for i in sorted(df_fitted_params['Participant ID'].unique())
    }


def compute_performance_metrics(df_choice: pd.DataFrame) -> pd.DataFrame:
    expected_reward_if_yes = - df_choice["price"]
    df_choice["expected_reward_if_correct"] = expected_reward_if_yes.clip(lower=0)
    df_choice["performance"] = expected_reward_if_yes * (1 - df_choice["action"])

    max_expected_reward_per_trial = df_choice["expected_reward_if_correct"].mean()
    perf = (df_choice.groupby(["participant_id", "Model"])["performance"].mean()/max_expected_reward_per_trial*100).copy()

    df_choice["Choice Accuracy"] = 0

    df_choice.loc[((df_choice["price"] < 0) & (df_choice["action"] == 0)), "Choice Accuracy"] = 1
    df_choice.loc[((df_choice["price"] > 0) & (df_choice["action"] == 1)), "Choice Accuracy"] = 1

    accuracy = (df_choice.groupby(["participant_id", "Model"], observed=False)["Choice Accuracy"].mean() * 100).copy()

    df_perf = pd.merge(left=perf, right=accuracy, left_on=["participant_id", "Model"], right_on=["participant_id", "Model"]).reset_index()
    return df_perf

# n_runs = 100
# model_classes = [DRA, FreqRA, StakesRA, EqualRA]
# dfs_perf = []

# for run in range(n_runs):
#     print(f"Run {run+1}/{n_runs}")

#     dfs_choice = []
#     for model_class in model_classes:
#         exp = ExperimentBestFitParams(model_class=model_class, params_and_data=get_params_and_data_for_model(model_class))
#         exp.run()
#         dfs_choice += [exp.extract_choice_data()]

#     df_choice = pd.concat(dfs_choice, ignore_index=True)
#     dfs_perf.append(compute_performance_metrics(df_choice))

# df_perf = pd.concat(dfs_perf, ignore_index=True)

# switch loops around for efficiency

n_runs = 10
model_classes = [DRA, FreqRA, StakesRA, EqualRA]
dfs_perf = []

for model_class in model_classes:
    
    params_and_data = get_params_and_data_for_model(model_class)

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        exp = ExperimentBestFitParams(model_class=model_class, params_and_data=params_and_data)
        exp.run()
        df_choice = exp.extract_choice_data()
        dfs_perf.append(compute_performance_metrics(df_choice))

df_perf = pd.concat(dfs_perf, ignore_index=True)


# format a little bit and fetch the real data
df_perf = df_perf.rename(columns={
    'participant_id': "id",
    'performance': 'model_performance',
    'Choice Accuracy': 'model_accuracy'
    })
data_perf = df.groupby(["id"])[["performance","accuracy"]].mean().reset_index()

# combine to put them together
perf = pd.merge(left=df_perf, right=data_perf, left_on="id", right_on="id")

# compute difference from data
perf["model_perf_diff"] = perf["model_performance"] - perf["performance"]
perf["model_acc_diff"] = perf["model_accuracy"] - perf["accuracy"]

# plot mean differences
sns.barplot(x='Model', y='model_perf_diff', data=perf)
plt.title('Mean Performance Difference from Data')
plt.xlabel('Model')
plt.ylabel('Absolute Difference')
plt.show()

# scatter plot
g = sns.lmplot(
    x='performance', 
    y='model_performance', 
    data=perf, 
    col='Model',
    height=6, 
    aspect=1,
    scatter_kws={'alpha': 0.5},
)

# Add diagonal line to each subplot
for ax in g.axes.flat:
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Model performance')
    ax.set_ylabel('Participant performance')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()