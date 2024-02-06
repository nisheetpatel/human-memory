import pandas as pd
from likelihood_new import ModelFitter
from model.model_new import DDRA, DEqualRA, DFreqRA, DStakesRA, Simulator, Task


# redefine results saver (change the actual class later)
class ResultsSaver:
    def __init__(self):
        self.all_results_df = pd.DataFrame()

    def store_all_results(self, all_results):
        print("Now storing results...")
        df = pd.DataFrame(
            [(result["id"], result["run_id"], result["agent_class"], result["fval"],
             *result["x"]) for result in all_results],
            columns=["Participant ID", "Run ID", "Model", "NLL", "lr_v", "lr_s", "lmda"]
        )
        self.all_results_df = pd.concat([self.all_results_df, df]).reset_index(drop=True)

# initialize the results saver
results_saver = ResultsSaver()

# things that don't need to be repeated
lmdas = [0.1, 0.25, 0.5, 1]
env = Task()
data_columns = ["sm_id", "price", "reward", "action"]
data = []

for lmda in lmdas:
    # generate data from each model
    agent_classes = [DDRA, DFreqRA, DStakesRA, DEqualRA]
    agents = [agent(lmda=lmda) for agent in agent_classes]
    simulators = [Simulator(env, agent, 4_000) for agent in agents]

    # collect data from each model
    for j, simulator in enumerate(simulators):
        data_tuples = simulator.train_agent(True)
        df = pd.DataFrame.from_records(data_tuples, columns=data_columns)
        df["id"] = f"{agent_classes[j].__name__}_{lmda}"
        data.append(df)

# model fitting
fitter = ModelFitter()
fitter.fit(results_saver, data)

# see results
print(results_saver.all_results_df)


########################################################################
# final results
########################################################################

# rename the dataframe and extract generated Model and params
df = results_saver.all_results_df
df["Model_gen"] = df["Participant ID"].apply(lambda x: x.split("_")[0])
df["lmda_gen"] = df["Participant ID"].apply(lambda x: x.split("_")[1])
df["lmda_gen"] = df["lmda_gen"].astype(float)

# model selection
idx = df.dropna().groupby(["Model_gen", "lmda_gen"])["NLL"].idxmin()
best_df = df.loc[idx].reset_index(drop=True)
best_df.replace({"DDRA": "DRA", "DFreqRA": "Frequency", "DStakesRA": "Stakes", "DEqualRA": "Equal Precision"}, inplace=True)

# get confusion matrix (think I used 0.04 <= lmda_gen < 0.12)
confusion_matrix = pd.crosstab(best_df["Model_gen"], best_df["Model"], normalize='index') * 100

# plot parameter and model recovery results
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

# plotting model recovery results
fig, ax = plt.subplots()
ax = sns.heatmap(confusion_matrix, fmt=".1f", annot=True, cbar=False)
plt.xlabel('Predicted Model')
plt.ylabel('Actual label')
plt.title('Model recovery')
plt.show()

# get data for parameter recovery plots
data = best_df.loc[(best_df["lmda"]<0.2) & (best_df["lmda_gen"]<0.2)]

# annotations (r and p values for parameter recovery plots)
annotations = []
models = ["DRA", "Equal Precision", "Frequency", "Stakes"]
for model in models:
    r, p = sp.stats.pearsonr(data.loc[data["Model_gen"]==model, "lmda_gen"],
                             data.loc[data["Model_gen"]==model, "lmda"])
    annotations.append(f"r={r:.2f}, p={p:.1e}")

# plotting parameter recovery results
fgrid = sns.lmplot(data=data, x="lmda_gen", y="lmda", hue="Model_gen", palette="deep")
plt.xlim([0.02, 0.16])
plt.ylim([0.0, 0.2])
ax = plt.gca()
colors = sns.color_palette("deep", 4)
for i, (text,model,color) in enumerate(zip(annotations,models,colors)):
    ax.text(0.05, 0.95-i*0.05, f"{model}: {text}", c=color, transform=ax.transAxes)
plt.show()

# plotting parameter recovery results
fgrid = sns.lmplot(data=data, x="lmda_gen", y="lmda", hue="Model_gen", palette="deep")
plt.xlim([0.02, 0.16])
plt.ylim([0.0, 0.2])
plt.xlabel("Actual memory cost ($\lambda$)")
plt.ylabel("Predicted memory cost ($\lambda$)")
ax = plt.gca()
colors = sns.color_palette("deep", 4)
for i, (text,model,color) in enumerate(zip(annotations,models,colors)):
    ax.text(0.05, 0.95-i*0.05, f"{model}: {text}", c=color, transform=ax.transAxes)
plt.show()


# plot correlation of accuracy with memory cost, learning rate of memory precision

# load data and get best fitting models and parameters for each participant
import numpy as np
from definitions import DATA_PATH

df = pd.read_csv(DATA_PATH+"all_fit_results_fixed_sb_cross-validated_new.csv")
df["log-lmda"] = np.log10(df["lmda"])
idx = df.dropna().groupby(["id"])["NLL"].idxmin()
best_df = df.loc[idx].reset_index(drop=True)


# correlation of accuracy with memory cost
r, p = sp.stats.pearsonr(best_df["accuracy"], best_df["log-lmda"])

sns.lmplot(data=best_df, y="accuracy", x="log-lmda", scatter_kws={"color": "darkgreen"}, 
           line_kws={"color": "darkgreen"})
# plt.xscale("log")
plt.xlabel("Memory cost ($\lambda$)")
plt.ylabel("Choice accuracy")

# Transform the axes for the text placement
plt.gca().text(0.5, 0.95, f"r={r:.2f}\np={p:.2e}", transform=plt.gca().transAxes, 
               horizontalalignment='center', verticalalignment='top')

plt.show()


# correlation of accuracy with learning rate of memory precision
r, p = sp.stats.pearsonr(best_df["accuracy"], best_df["lr_s"])

sns.lmplot(data=best_df, y="accuracy", x="lr_s")
plt.xlabel("Learning rate for memory precision ($\sigma$)")
plt.ylabel("Choice accuracy")

# Transform the axes for the text placement
plt.gca().text(0.5, 0.95, f"r={r:.2f}\np={p:.2e}", transform=plt.gca().transAxes, 
               horizontalalignment='center', verticalalignment='top')

plt.show()