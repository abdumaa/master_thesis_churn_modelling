import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.special import expit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import seaborn as sns
# import scikitplot as skplt
import matplotlib.pyplot as plt

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
from churn_modelling.preprocessing.mrmr import mrmr
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.modelling.custom_loss import FocalLoss, WeightedLoss
from churn_modelling.utils.mem_usage import reduce_mem_usage
from churn_modelling.preprocessing.mrmr import _correlation_scorer

df_temp = pd.read_csv('../data/toydata.csv', index_col=0)

df = df_temp.copy()

df_corr = _correlation_scorer(df)
# MRMR
iterations_df = mrmr(df, target="storno", cv=5)
iterations_df["ITERATION"] = iterations_df["ITERATION"].astype(int)
iterations_df["MRMR_SCORE"] = (
    iterations_df["MRMR_SCORE"].replace(np.inf, 5000000.0).replace("", 0.0)
)
iterations_df.head(13)
sns.relplot(
    x="ITERATION",
    y="MRMR_SCORE",
    dashes=False,
    markers=True,
    kind="line",
    data=iterations_df,
)

# Reduce mem usage
df = reduce_mem_usage(df)

# Split X and y
X = df.iloc[:, 1:]
y = df["storno"]

# Transform df categorical dtypes for lgbm
X, cat_features = to_categorical(df=X)

# # Impute missings with mean
# imp = SimpleImputer(strategy="mean")
# imp.fit(X)
# X_imputed = imp.transform(X)
# X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
# X_imputed.head()


# Train, val, test split
X_train, X_test, y_train, y_test = split_train_test(X=X, y=y)
X_train, X_val, y_train, y_val = split_train_test(X=X_train, y=y_train)

# Downsample Training set
# X_train["storno_corrected"] = y_train
# df_ds_train = resample(X_train, y="storno_corrected")
# X_train = df_ds_train[features]
# y_train = df_ds_train["storno_corrected"]

# len(y_train[y_train == 1]) / len(y_train)


### lgbm


# Learning rate shrinkage for better convergence
def learning_rate_decay(current_iter, base=0.1, decay_factor=0.99):
    """Define learning decay shrinkage for better and faster convergence."""
    base_learning_rate = base
    lr = base_learning_rate * np.power(decay_factor, current_iter)
    return lr if lr > 1e-3 else 1e-3


fl = FocalLoss(gamma=3, alpha=0.55)

lgbm = lgb.LGBMClassifier(
    boosting_type="gbdt",  # also try 'dart' and 'goss'
    objective="binary",  # or fl.lgb_obj
    max_depth=-1,
    # class_weight="balanced",
    n_jobs=-1,
    n_estimators=5000,
    random_state=1,
    silent=True,
    importance_type="split",
)
fit_params = {  # Val set used for early stopping criterion
    "early_stopping_rounds": 30,
    # "eval_class_weight": "balanced",
    "eval_metric": "logloss",  # or fl.lgb_eval
    "eval_set": [(X_val, y_val)],
    # "init_score": np.full_like(y_train, fl.init_score(y_train), dtype=float), # use y_val for initializer because y_train is downsampled # noqa
    "callbacks": [lgb.reset_parameter(learning_rate=learning_rate_decay)],
    "verbose": 100,
    # "categorical_feature": cat_features,
    # "monotonicity"
}
params = {
    "num_leaves": sp_randint(6, 50),  # Max number of leaves in base learner
    # "max_depth": [-1],  # Max depth for base learners
    # "learning_rate": [0.1],
    # "n_estimators": [100],  # Number of trees
    # "subsample_for_bin ": [200000],  # Number of samples for constructing bins (200000 default value) # noqa
    # "min_split_gain": [0.0],  # Minimum loss reduction required to make a further partition on a leaf node of the tree # noqa
    "min_child_weight": [
        1e-5,
        1e-2,
        1e-1,
        1,
        1e1,
        1e4,
    ],  # Minimum sum of instance weight (hessian) needed in a child (leaf) # noqa
    "min_child_samples": sp_randint(
        100, 500
    ),  # Minimum number of data needed in a child (leaf) # noqa
    "subsample": sp_uniform(
        loc=0.4, scale=0.6
    ),  # Subsample ratio of the training instance. low in baseline model. Increase later to 1.0!!! # noqa
    # "subsample_freq": [0],  # Frequency of subsample, <=0 means no enable. Every k iteration bagging is used # noqa
    "colsample_bytree": sp_uniform(
        loc=0.6, scale=0.4
    ),  # Subsample ratio of columns when constructing each tree, lower means less computation time, (overfitting) higher means hgher accuracy? # noqa
    "reg_alpha": [0, 1, 5, 10, 100],
    "reg_lambda": [0, 1, 5, 10, 100],
}
# scorers = {
#     "Precision": make_scorer(precision_score),
#     "Recall": make_scorer(recall_score),
#     "Accuracy": make_scorer(accuracy_score),
# }
scorer = make_scorer(
    average_precision_score,
    greater_is_better=True,
    needs_proba=True,
    average="weighted",
)
lgbm_rsc = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=params,
    n_iter=10,
    # scoring=scorer, # using estimators scoring
    random_state=43,
    n_jobs=-1,
    cv=3,
    # refit="Accuracy", set to default ("True")
    # return_train_score=True,
    verbose=True,
)
lgbm_fit = lgbm_rsc.fit(X_train, y_train, **fit_params)
print(
    "Best score reached: {} with params: {} ".format(
        lgbm_fit.best_score_, lgbm_fit.best_params_
    )
)
# for custom loss
predictions = expit(fl.init_score(y_train) + lgbm_fit.predict(X_test))
predictions = (predictions >= 0.5).astype("int")
preds = expit(fl.init_score(y_train) + lgbm_fit.predict_proba(X_test))
acc = accuracy_score(y_test, predictions)  #
prec = precision_score(y_test, predictions)  # TP/(TP + FP)
rec = recall_score(y_test, predictions)  # TP/(TP + FN)
confusion_matrix(y_test, predictions)  # 0 TP, 0 FP!

# without custom loss
predictions = lgbm_fit.predict(X_test)
preds = lgbm_fit.predict_proba(X_test)
pred_list = [p[1] for p in preds]
predmod = [0 if p < 0.8 else 1 for p in pred_list]  # (pred_list >= 0.5).astype("int")
acc = accuracy_score(y_test, predictions)  # 0.945
prec = precision_score(y_test, predictions)  # 0.9189 TP/(TP + FP)
rec = recall_score(y_test, predictions)  # 0.395 TP/(TP + FN)
confusion_matrix(y_test, predictions)
confusion_matrix(y_test, predmod)


# visualizations

sns.distplot(pred_list)

skplt.metrics.plot_roc_curve(y_test, preds)
plt.show()

# skplt.metrics.plot_precision_recall_curve(y_test, preds)
# plt.show()

feat_imp = pd.Series(
    lgbm_fit.best_estimator_.feature_importances_, index=X_train.columns
)
feat_imp.nlargest(20).plot(kind="barh", figsize=(8, 10))