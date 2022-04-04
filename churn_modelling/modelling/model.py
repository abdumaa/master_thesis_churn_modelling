import lightgbm as lgb
import pandas as pd

from lgbm import LGBM

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


def create_lgbm_fit():  # maybe include dim_reduction as param here?
    """Create Fit with LGBMClassifier."""

    # LGBM parameter dictionaries
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
        "n_iter": 10,
        "random_state": 43,
        "n_jobs": -1,
        "cv": 3,
        "verbose": 100,
    }

    # Load raw flatfile
    df_temp = pd.read_csv('../data/toydata.csv', index_col=0)
    df = df_temp.copy()

    model_pl = LGBM(df=df, target="storno", test_size=0.1)
    df_train, df_val, df_test = model_pl.create_train_val_test()
    df_ds_train = model_pl.create_sampling(df_to_sample=df_train, frac=0.1)

    # MRMR Option

    feature_set = model_pl.get_featureset_from_latest_run()

    return model_pl.fit_lgbm(
        df_train=df_ds_train,
        df_val=df_val,
        hp_fix_dict=hp_fix_dict,
        hp_tune_dict=hp_tune_dict,
        hp_eval_dict=hp_eval_dict,
        rscv_params=rscv_params,
        feature_set=feature_set,
        reduce_df_mem=True,
        save_model=True,
        learning_rate_decay=True,
        focal_loss=None,  # FocalLoss(alpha=0.6, gamma=2)
    )
