import pandas as pd
import numpy as np
import ast
import glob
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from scipy.special import expit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import seaborn as sns
# import scikitplot as skplt
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
)

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.mrmr import mrmr, _lgbm_scorer
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.modelling.custom_loss import FocalLoss, WeightedLoss
from churn_modelling.modelling.ebm import EBM
from churn_modelling.utils.mem_usage import reduce_mem_usage
from churn_modelling.preprocessing.mrmr import _correlation_scorer

# Load data
df_temp = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
df = df_temp.copy()

# Load EBM
model_ebm = EBM(df=df, target="churn", test_size=0.1)

# Train Test Split
df_train, df_test = model_ebm.create_train_test()

# Downsample Training Set
X_us, y_us = model_ebm.create_sampling(df_to_sample=df, sampling="smote", frac="balanced")

# Split features into quotation and fix_features
quot_feats, fix_feats = model_ebm.split_quotation_fix_features()

# MRMR
iterations_df = model_ebm.feature_selection(
    df_to_dimreduce=df_train,
    variable_names=quot_feats,
    cv=5,
)
model_ebm.visualize_feature_selection(iterations_df)

# Dimension reduction
#feature_set = ast.literal_eval(iterations_df["SELECTED_SET"][20])
fix_feats.extend([
    "storno",
    "n_requests_1",
    "diff_avg_vjnbe_requests_3",
    "diff_n_requests_3",
    "other_hsntsn_requests_3",
])
df_train = df_train[fix_feats]
df_test = df_test[fix_feats]


### EBM
hp_fix_dict = {
    "validation_size": 0.1111,
    "early_stopping_rounds": 30,
    "early_stopping_tolerance": 1e-4,
    "max_rounds": 5000,
}
hp_tune_dict = {
    "interactions": sp_randint(5, 10),
    "outer_bags": sp_randint(20, 30),
    "inner_bags": sp_randint(20, 30),
    "learning_rate": sp_uniform(loc=0.009, scale=0.006),
    "min_samples_leaf": sp_randint(2, 10),
    "max_leaves": sp_randint(2, 5),
}
rscv_params = {
    "n_iter": 100,
    "random_state": 43,
    "n_jobs": -1,
    "cv": 3,
    "verbose": 100,
}

seed = 12

ebm = model_ebm.fit_ebm(
    df_train=df_train,
    hp_fix_dict=hp_fix_dict,
    hp_tune_dict=hp_tune_dict,
    rscv_params=rscv_params,
    seed=seed,
)

# ebm = ExplainableBoostingClassifier(
#     interactions=10,
#     outer_bags=25, # default=8
#     inner_bags=25, # default=0
#     learning_rate=0.01,
#     validation_size=0.1111,
#     early_stopping_rounds=30,
#     early_stopping_tolerance=1e-4,
#     max_rounds=5000,
#     min_samples_leaf=2,
#     max_leaves=3,
#     n_jobs=-1,
#     random_state=seed,
# )

y_train = df_train["storno"]
X_train = df_train.drop(["storno"], axis=1)

ebm_fit = ebm.fit(X_train, y_train)

### Predict
y_true = df_test["storno"]
X_test = df_test.drop(["storno"], axis=1)

preds = ebm_fit.predict(X_test)


### Evaluation
acc = accuracy_score(y_true, preds)  # 0.9915 / 0.9905
prec = precision_score(y_true, preds)  # 0.8462 / 0.7333 TP/(TP + FP)
rec = recall_score(y_true, preds)  # 0.2136 / 0.1078 TP/(TP + FN)
confusion_matrix(y_true, preds)

### Explanations
ebm_global = ebm.explain_global()
show(ebm_global)


### Save Model
time = datetime.now().strftime("%y%m%d%H%M%S")
dump(
    ebm_fit,
    f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/ebm_fit_{time}.joblib",  # find solution for that # noqa
)

### Load Model
list_of_fits = glob.glob(
    "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/*"  # find solution for that # noqa
)
latest_file = max(list_of_fits, key=os.path.getctime)
last_part = latest_file.rsplit("ebm_fit_", 1)[1]
ebm_fit_new = load(latest_file)

preds = ebm_fit_new.predict(X_test)