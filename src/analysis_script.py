from analysis.data import Processor as DataProcessor
from analysis.hierarchical import Classifier, HierarchicalModel, Plotter
from definitions import DATA_PATH, FIGURE_PATH, MODEL_PATH

# set parameters for fitting hierarchical model
hierarchical_model_params = {"n_chains": 4, "n_samples": 2_000}

# set parameters for classifier
classifier_params = {"equality_thresh": 1.7, "class_thresh": 20}

# read and process data, then extract performance metrics
data_processor = DataProcessor(path=DATA_PATH)
df = data_processor.get_processed_data()
df_good = df.loc[df["above_chance"]]
perf = data_processor.extract_performance_metrics(df)

# fit hierarchical model to good subjects
model = HierarchicalModel()
choice_data_dict = model.get_choice_data_dict(df_good)
fit = model.fit_posterior(choice_data_dict, **hierarchical_model_params)
model.save(fit, f"{MODEL_PATH}good_subjects.pkl")

# classify subjects
classifier = Classifier(fit, **classifier_params)
df_class = classifier.classify()
perf = classifier.merge_with_performance_metrics(perf)

# plot posterior at individual level
plotter = Plotter(fit, save_path=FIGURE_PATH)
plotter.plot_posterior_over_betas_individual("betas_good-subjects", perf)
plotter.plot_posteriors_at_group_level("group_good-subjects")

# split participants by their assigned class
model_names = ["DRA", "Freq", "Stakes", "EP"]
ids = [classifier.get_ids(model_name) for model_name in model_names]
dfs = [df_good.loc[df_good["id"].isin(ids_model)].copy() for ids_model in ids]

# fit hierarchical model to good subjects of each class
fits = []
for i, (df, model_name) in enumerate(zip(dfs, model_names)):
    print(f"Fitting model {model_name} of {len(dfs)} subjects")
    model = HierarchicalModel()
    choice_data_dict = model.get_choice_data_dict(df)
    fit = model.fit_posterior(choice_data_dict, **hierarchical_model_params)
    fits.append(fit)
    model.save(fit, f"{MODEL_PATH}{model_name}.pkl")
