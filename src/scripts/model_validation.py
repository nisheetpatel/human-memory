import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from src.data.processor import get_processed_data
from src.definitions import DATA_PATH
from src.simulation.experiment import ExperimentBestFitParams, RAModelParams
from src.simulation.models import DRA, EqualRA, FreqRA, StakesRA
from src.simulation.simulator import Simulator

########################################################################
# get the best-fitting parameters for each subject
########################################################################

# load the parameters
df_fitted_params = pd.read_csv(DATA_PATH + "all_fit_results_fixed_sb_cross-validated_new.csv")

# function to extract best-fitting parameters for each subject/model
def get_best_params(participant_id: int, model: str, df: pd.DataFrame = df_fitted_params) -> RAModelParams:
    """
    Extract the best fitting parameters for a given participant and model.

    Args:
        df (pd.DataFrame): The DataFrame containing the fitted parameters.
        participant_id (int): The ID of the participant.
        model (str): The name of the model.

    Returns:
        RAModelParams: An instance of RAModelParams with the best fitting parameters.
    """
    # Filter the DataFrame for the specific participant and model
    participant_data = df[(df['Participant ID'] == participant_id) & (df['Model'] == model)]

    if participant_data.empty:
        raise ValueError(f"No data found for Participant ID {participant_id} and Model {model}")

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


model_class = DRA
exp = ExperimentBestFitParams(model_class=model_class, params_and_data=get_params_and_data_for_model("DDRA"))
exp.run()

df_choice = exp.extract_choice_data()


# analyze