import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

### Import data set
df = pd.read_csv('toydata.csv', index_col=0)
df_oop = pd.read_csv('toydata_oop.csv', index_col=0)
df_oos = pd.read_csv('toydata_oos.csv', index_col=0)

### EDA Plots
# Set up figure
fig = plt.figure(figsize = (20,25))

# Violinplots
nbin_var = df[['n_requests_1', 'n_requests_2', 'n_requests_3',
       'diff_n_requests_1', 'diff_n_requests_2', 'diff_n_requests_3',
       'diff_avg_vjnbe_requests_1', 'diff_avg_vjnbe_requests_2',
       'diff_avg_vjnbe_requests_3', 'other_hsntsn_requests_1',
       'other_hsntsn_requests_2', 'other_hsntsn_requests_3', 'n_accident',
       'sum_accident_cost', 'vehicle_age', 'diff_car_holder',
       'contract_age_months', 'age_contract_holder', 'age_youngest_driver',
       'years_driving_license']]
for i, v in enumerate(nbin_var):
    axes = fig.add_subplot(6, 4, i+1)
    b = sns.violinplot(x='churn', y=v, data=df, ax=axes)
    b.set_xlabel("Churn", fontsize=12)
    b.set_ylabel(v, fontsize=12)

# Save fig
fig.savefig('../../tex/images/violin.png', dpi=300, bbox_inches="tight")


# Quick stats
df.groupby("churn").mean()
df.groupby("churn").std()
df.groupby("churn").median()

### Baseline metrics
# Naive B1: Never predict churn
b1_preds = np.zeros(len(df_oos))
b1_conf_oos = confusion_matrix(df_oos["churn"], b1_preds)
b1_conf_oop = confusion_matrix(df_oop["churn"], b1_preds)

# Naive B2: Predict churn when n_requests_1 > 0
b2_preds_oos = np.where(df_oos['n_requests_1']>0, 1, 0)
b2_preds_oop = np.where(df_oop['n_requests_1']>0, 1, 0)
b2_conf_oos = confusion_matrix(df_oos["churn"], b2_preds_oos)
b2_conf_oop = confusion_matrix(df_oop["churn"], b2_preds_oos)

# Scores
accuracy_score(df_oop["churn"], b2_preds_oop)
precision_score(df_oop["churn"], b2_preds_oop)
recall_score(df_oop["churn"], b2_preds_oop)
f1_score(df_oop["churn"], b2_preds_oop)
roc_auc_score(df_oop["churn"], b2_preds_oop)
average_precision_score(df_oop["churn"], b2_preds_oop)

### Vizualization
ax = sns.heatmap(
    b2_conf_oop,#/np.sum(conf_matrix),
    annot=True,
    #fmt='.2%',
    cmap='Blues',
    fmt="d",
)

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Save the visualization of the Confusion Matrix.
sns_plot = ax.get_figure()
sns_plot.savefig('../../tex/images/conf_b2_oop.png', bbox_inches='tight')