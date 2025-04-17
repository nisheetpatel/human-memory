import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.processor import (
    DataProcessor,
    compute_performance_metrics,
    get_processed_data,
)
from src.definitions import DATA_PATH, FIGURE_PATH

# read and process data, then extract performance metrics
df = get_processed_data(data_path=DATA_PATH + "pilot_slot-machines_3/")
perf = compute_performance_metrics(df)

# get null distribution from data processor for plots
processor = DataProcessor(path=DATA_PATH + "pilot_slot-machines_3/")
df = processor.get_processed_data()
perf_null = processor._get_null_distribution(df)


########################################################
# Joint distribution of performance vs. accuracy
########################################################

plt.figure(figsize=(6, 6))
sns.jointplot(data=perf, x='accuracy', y='performance', kind='reg')
# plt.title('Performance vs. Accuracy')
plt.xlabel('Accuracy (%)', fontsize=14)
plt.ylabel('Performance (%)', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURE_PATH + 'performance_vs_accuracy.png', dpi=300)
plt.savefig(FIGURE_PATH + 'performance_vs_accuracy.svg')
plt.close()


########################################################
# Performance: Subjects vs Null
########################################################

perf_threshold = np.percentile(perf_null, 95)

# Create a dataframe for the subjects' data
df_subjects = pd.DataFrame({
    'Performance': perf['performance'],
    'Group': np.where(perf['above_chance'], 'Above Chance', 'Below Chance')
})

# Create a dataframe for the null distribution
df_null = pd.DataFrame({
    'Performance': perf_null,
    'Group': 'Null'
})

# Combine the dataframes
df_combined = pd.concat([df_subjects, df_null])

# Set up the plot
plt.figure(figsize=(6, 6))
sns.set_style("whitegrid")

# Create violinplots
sns.violinplot(x='Group', y='Performance', data=df_combined, hue='Group',
               order=['Null', 'Below Chance', 'Above Chance'],
               palette={'Null': 'lightgrey', 'Below Chance': 'lightcoral', 'Above Chance': 'lightblue'},
               cut=0)#, scale='width', inner=None)

# Overlay individual points for subjects
sns.stripplot(x='Group', y='Performance', data=df_subjects, hue='Group',
              order=['Below Chance', 'Above Chance'],
              palette={'Below Chance': 'firebrick', 'Above Chance': 'royalblue'},
              size=4, jitter=True, alpha=0.6)

# Text overlay to display number of subjects
plt.text(1, perf_threshold + 3, f'N = {len(perf.loc[~perf["above_chance"]])}', 
         ha='center', va='center', color='firebrick', fontweight='bold', fontsize=12)
plt.text(2, perf_threshold - 3, f'N = {len(perf.loc[perf["above_chance"]])}', 
         ha='center', va='center', color='royalblue', fontweight='bold', fontsize=12)


# Add the 95th percentile line
plt.axhline(y=perf_threshold, color='black', linestyle='--', linewidth=2)
plt.text(0, perf_threshold + 3, f'95th percentile: {perf_threshold:.1f}', 
         ha='center', va='center', color='black', fontsize=12)

# Customize the plot
plt.title('Performance Distribution: Subjects vs Null', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.ylim(-25, max(df_combined['Performance'].max(), 100))

# Save and show the plot
plt.tight_layout()
plt.savefig(FIGURE_PATH + 'performance_dist_subjects_vs_null.png', dpi=300)
plt.savefig(FIGURE_PATH + 'performance_dist_subjects_vs_null.svg')
plt.close()