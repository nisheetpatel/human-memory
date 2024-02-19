import numpy as np
import pandas as pd

from analysis.data import Processor as DataProcessor
from analysis.hierarchical import HierarchicalModel, Plotter
from definitions import DATA_PATH, FIGURE_PATH, MODEL_PATH

# import subjects classified by BADS and BHLR testing
df_classes = pd.read_csv(DATA_PATH+'model_classes.csv')
df_classes = df_classes.loc[:, ["id", "class", "class_bads"]]

# set parameters for fitting hierarchical model
hierarchical_model_params = {"n_chains": 4, "n_samples": 25_000}

# read and process data, then extract performance metrics
data_processor = DataProcessor(path=DATA_PATH+"pilot_slot-machines_3/")
df = data_processor.get_processed_data()
df_good = df.loc[df["above_chance"]]
df_good = df_good.merge(df_classes, left_on="id", right_on="id")
perf = data_processor.extract_performance_metrics(df)

# define model classes for BADS
model_classes = df_classes["class_bads"].unique()[:-1]

# fit hierarchical model to each class of subjects classified by BADS
for model_class in model_classes:
    model = HierarchicalModel()

    # extract models classified as model_class and reset ids
    df_model = df_good.loc[df_good["class_bads"] == model_class].copy()
    id_map = dict(zip(
        df_model['id'].unique(),
        np.arange(1, len(df_model['id'].unique()) + 1)
    ))
    df_model["id"] = df_model["id"].map(id_map)

    # BHLR fits
    choice_data_dict = model.get_choice_data_dict(df_model)
    fit = model.fit_posterior(choice_data_dict, **hierarchical_model_params)
    model.save(fit, f"{MODEL_PATH}{model_class}_BADS.pkl")

    # plot posterior at individual level
    plotter = Plotter(fit, save_path=FIGURE_PATH)
    plotter.plot_posterior_over_betas_individual(f"betas-individual_{model_class}_BADS")
    plotter.plot_posteriors_at_group_level(f"betas-group_{model_class}_BADS")

# fit hierarchical model to each class of subjects classified by BHLR test
model_classes = df_classes["class"].unique()

for model_class in model_classes:
    model = HierarchicalModel()

    # extract models classified as model_class and reset ids
    df_model = df_good.loc[df_good["class"] == model_class].copy()
    id_map = dict(zip(
        df_model['id'].unique(),
        np.arange(1, len(df_model['id'].unique()) + 1)
    ))
    df_model["id"] = df_model["id"].map(id_map)

    # BHLR fits
    choice_data_dict = model.get_choice_data_dict(df_model)
    fit = model.fit_posterior(choice_data_dict, **hierarchical_model_params)
    model.save(fit, f"{MODEL_PATH}{model_class}_BHLR.pkl")

    # plot posterior at individual level
    plotter = Plotter(fit, save_path=FIGURE_PATH)
    plotter.plot_posterior_over_betas_individual(f"betas-individual_{str(model_class)}_BHLR")
    plotter.plot_posteriors_at_group_level(f"betas-group_{str(model_class)}_BHLR")


# Parallelized code (somehow not working because asyncio issues with Stan)

# def fit_and_plot_for_model_class(model_class, classification_method):
#     model = HierarchicalModel()

#     # extract models classified as model_class and reset ids
#     df_model = df_good.loc[df_good[classification_method] == model_class].copy()
#     id_map = dict(zip(
#         df_model['id'].unique(),
#         np.arange(1, len(df_model['id'].unique()) + 1)
#     ))
#     df_model["id"] = df_model["id"].map(id_map)

#     # BHLR fits
#     choice_data_dict = model.get_choice_data_dict(df_model)
#     fit = model.fit_posterior(choice_data_dict, **hierarchical_model_params)
#     model.save(fit, f"{MODEL_PATH}{model_class}_{classification_method}.pkl")

#     # plot posterior at individual level
#     plotter = Plotter(fit, save_path=FIGURE_PATH)
#     plotter.plot_posterior_over_betas_individual(f"betas-individual_{model_class}_{classification_method}")
#     plotter.plot_posteriors_at_group_level(f"betas-group_{model_class}_{classification_method}")


# # Function to pass to the multiprocessing Pool
# def process_model_class(args):
#     return fit_and_plot_for_model_class(*args)


# if __name__ == "__main__":

#     # import subjects classified by BADS and BHLR testing
#     df_classes = pd.read_csv(DATA_PATH+'model_classes.csv')
#     df_classes = df_classes[:-1]
#     df_classes = df_classes.loc[:, ["id", "class", "class_bads"]]

#     # set parameters for fitting hierarchical model
#     hierarchical_model_params = {"n_chains": 4, "n_samples": 5_000}

#     # read and process data, then extract performance metrics
#     data_processor = DataProcessor(path=DATA_PATH+"pilot_slot-machines_3/")
#     df = data_processor.get_processed_data()
#     df_good = df.loc[df["above_chance"]]
#     df_good = df_good.merge(df_classes, left_on="id", right_on="id")
#     perf = data_processor.extract_performance_metrics(df)

#     # For BADS classification
#     with ThreadPoolExecutor() as executor:
#         executor.map(process_model_class, [(model_class, "class_bads") for model_class in df_classes["class_bads"].unique()])

#     # For BHLR classification
#     with ThreadPoolExecutor() as executor:
#         executor.map(process_model_class, [(model_class, "class") for model_class in df_classes["class"].unique()])
    