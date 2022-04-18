import pandas as pd
import numpy as np
import ast
import glob
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from scipy.special import expit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import seaborn as sns
# import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
)

import lightgbm as lgb

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.mrmr import mrmr, _lgbm_scorer
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.modelling.custom_loss import FocalLoss, WeightedLoss
from lgbm import LGBM
from churn_modelling.utils.mem_usage import reduce_mem_usage
from churn_modelling.preprocessing.mrmr import _correlation_scorer


# Load data
df_trainval_temp = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
df_trainval = df_trainval_temp.copy()

# Load LGBM
model_pl = LGBM(df=df_trainval, target="churn", test_size=0.1)

# Train Test Split
df_train, df_val = model_pl.create_train_val()

# Downsample Training Set
df_ds_train = model_pl.create_sampling(df_to_sample=df_train, frac=0.1)

# MRMR and Dimension Reduction
best_feats = model_pl.get_best_quot_features(df_to_dimreduce=df_train)
df_train = df_train[best_feats]
df_val = df_val[best_feats]
df_test = df_test[best_feats]


### lgbm

hp_fix_dict = {
    "objective": "binary",
    "max_depth": -1,
    "n_estimators": 1000,
    "random_state": 1,
    "importance_type": "split",
}
hp_tune_dict = {
    "num_leaves": sp_randint(6, 50),
    "min_child_weight": [1e-5, 1e-2, 1e-1, 1, 1e1, 1e4],
    "min_child_samples": sp_randint(100, 500),
    "subsample": sp_uniform(loc=0.4, scale=0.6),
    "colsample_bytree": sp_uniform(loc=0.6, scale=0.4),
    "reg_alpha": [0, 1, 5, 10, 100],
    "reg_lambda": [0, 1, 5, 10, 100],
}
hp_eval_dict = {
    "eval_metric": "logloss",
    "callbacks": [lgb.log_evaluation(100), lgb.early_stopping(30)],
}
rscv_params = {
    "n_iter": 100,
    "random_state": 43,
    "n_jobs": -1,
    "cv": 3,
    "verbose": 100,
}

lgbm_fit = model_pl.fit_lgbm(
    df_train=df_train,
    df_val=df_val,
    hp_fix_dict=hp_fix_dict,
    hp_tune_dict=hp_tune_dict,
    hp_eval_dict=hp_eval_dict,
    rscv_params=rscv_params,
    feature_set=None,
    reduce_df_mem=True,
    save_model=True,
    learning_rate_decay=True,
    focal_loss=None,  # FocalLoss(alpha=0.6, gamma=2)
)

### Predict
y_true = df_test["storno"]
X_test = df_test.drop(["storno"], axis=1)

preds, preds_proba = model_pl.predict(
    X=X_test,
    predict_from_latest_fit=False,
    lgbm_fit=lgbm_fit,
)

pred6 = [0 if p < 0.6 else 1 for p in preds_proba]  # (pred_list >= 0.5).astype("int")
acc = accuracy_score(y_true, preds)  # 0.9911
prec = precision_score(y_true, preds)  # 0.7692 TP/(TP + FP)
rec = recall_score(y_true, preds)  # 0.194 TP/(TP + FN)
confusion_matrix(y_true, preds)


# Get latest lgbm model
list_of_fits = glob.glob(
    "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/*"  # find solution for that # noqa
)
latest_file = max(list_of_fits, key=os.path.getctime)
last_part = latest_file.rsplit("lgbm_fit_", 1)[1]
lgbm_fit = load(latest_file)



# visualizations
y_train = df_train["storno"]
X_train = df_train.drop(["storno"], axis=1)
PartialDependenceDisplay.from_estimator(lgbm_fit, X_train, ["years_driving_license"], target=y_true)

# skplt.metrics.plot_roc_curve(y_test, preds)
# plt.show()

# skplt.metrics.plot_precision_recall_curve(y_test, preds)
# plt.show()

feat_imp = pd.Series(
    lgbm_fit.best_estimator_.feature_importances_, index=X_ds_train.columns
)
feat_imp.nlargest(20).plot(kind="barh", figsize=(8, 10))

