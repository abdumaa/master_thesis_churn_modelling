import pandas as pd
import numpy as np
import ast
import glob
from joblib import dump, load
import seaborn as sns
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from churn_modelling.modelling.ebm import EBM

models = [
    'ebm_none_best_quot',
    'ebm_none_no_quot',
    'ebm_down1_best_quot',
    'ebm_down1_no_quot',
    'ebm_down2_best_quot',
    'ebm_down2_no_quot',
    'ebm_down3_best_quot',
    'ebm_down3_no_quot',
    'ebm_up1_best_quot',
    'ebm_up1_no_quot',
    'ebm_up2_best_quot',
    'ebm_up2_no_quot',
    'ebm_up3_best_quot',
    'ebm_up3_no_quot',
    'ebm_smote_best_quot',
    'ebm_smote_no_quot',
]

# Load data
df_train = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
df_oos = pd.read_csv('../data/toydata_oos.csv', index_col=0)
df_oop = pd.read_csv('../data/toydata_oop.csv', index_col=0)

# Load EBM
class_ebm = EBM(df=df_train, target="churn", test_size=0.1)

# Load fitted Model
ebm_fit = load(
    "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/ebm_fit_ebm_down2_best_quot.joblib"
)

# Extract feature set used for modelling
single_feats, interaction_feats = class_ebm.get_featureset_from_fit(
    ebm_fit,
    include_target=False,
    include_interactions=True
)

# Split y and X
y_train = df_train["churn"]
X_train = df_train.drop(["churn"], axis=1)[single_feats]
y_oos = df_oos["churn"]
X_oos = df_oos.drop(["churn"], axis=1)[single_feats]
y_oop = df_oop["churn"]
X_oop = df_oop.drop(["churn"], axis=1)[single_feats]

### Predict
preds_oos = ebm_fit.predict(X_oos)
preds_oop = ebm_fit.predict(X_oop)

### Evaluation
# OOS
acc_oos = accuracy_score(y_oos, preds_oos)
prec_oos = precision_score(y_oos, preds_oos)
rec_oos = recall_score(y_oos, preds_oos)
f1_oos = f1_score(y_oos, preds_oos)
auroc_oos = roc_auc_score(y_oos, preds_oos)
auprc_oos = average_precision_score(y_oos, preds_oos)

# OOP
acc_oop = accuracy_score(y_oop, preds_oop)
prec_oop = precision_score(y_oop, preds_oop)
rec_oop = recall_score(y_oop, preds_oop)
f1_oop = f1_score(y_oop, preds_oop)
auroc_oop = roc_auc_score(y_oop, preds_oop)
auprc_oop = average_precision_score(y_oop, preds_oop)

### Explanations
ebm_global = ebm_fit.explain_global()
show(ebm_global)


idx = ebm_global.selector.index[ebm_global.selector['Name'] == 'diff_avg_vjnbe_requests_3 x n_accident'][0]
data_dict = ebm_global.data(idx)
fig = ebm_global.visualize(0)
fig.write_image("test.png")
x_vals = data_dict["names"].copy()
y_vals = data_dict["scores"].copy()
y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]
x = np.array(x_vals)
sns.lineplot(x,y_vals, drawstyle='steps-post')
plt.xlabel('n_requests_1')
plt.ylabel('$\mathregular{k_{1}}$(n_requests_1)')
plt.savefig('../../tex/images/ebm_nrequests.png')