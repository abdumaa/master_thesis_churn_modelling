import pandas as pd

import numpy as np
import ast
from datetime import datetime
import glob
import os
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV

from scipy.special import expit
import seaborn as sns

import lightgbm as lgb

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.mrmr import mrmr
from churn_modelling.utils.mem_usage import reduce_mem_usage

empty_df = pd.DataFrame()


class LGBM:
    """Class for lgbm modelling and predictions for probabilities of midcontract churns."""

    def __init__(self, df, target, test_size=0.1):
        self.df = df
        self.target = target
        self.test_size = test_size

    def create_train_val_test(self):
        """Split train, val and test."""
        train, test = split_train_test(
            df=self.df, target=self.target, test_size=self.test_size
        )
        train, val = split_train_test(
            df=train,
            target=self.target,
            test_size=self.test_size / (1 - self.test_size),
        )
        return train, val, test

    def create_sampling(self, df_to_sample=None, sampling="down", frac="balanced"):
        """Create (synthetic) up- or down-sampling."""
        if df_to_sample is not None:
            return resample(
                df_to_sample=df_to_sample, y=self.target, sampling=sampling, frac=frac
            )
        else:
            return resample(
                df_to_sample=self.df, y=self.target, sampling=sampling, frac=frac
            )

    def feature_selection(self, df_to_dimreduce=None, cv=5, sample=False):
        """Create feature_selection_df using MRMR-Scorer."""
        if df_to_dimreduce is not None:
            return mrmr(
                df_to_dimreduce=df_to_dimreduce,
                target=self.target,
                cv=cv,
                sample=sample,
            )
        else:
            return mrmr(
                df_to_dimreduce=self.df, target=self.target, cv=cv, sample=sample
            )

    def visualize_feature_selection(self, feature_selection_df):
        """Visualizes the log MRMR-Score of each iteration based on output of MRMR."""
        return sns.relplot(
            x="ITERATION",
            y="LOG_MRMR_SCORE",
            dashes=False,
            markers=True,
            kind="line",
            data=feature_selection_df,
        )

    def get_featureset_from_mrmrdf(
        self, feature_selection_df, iteration, include_target=True
    ):
        """Select feature set based on output of MRMR and iteration."""
        feature_set = ast.literal_eval(feature_selection_df["SELECTED_SET"][iteration])
        feature_set.append(self.target)

        return feature_set

    def get_featureset_from_latest_run(self, include_target=True):
        """Select feature set based on set of latest run."""
        list_of_fits = glob.glob(
            "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/*"  # find solution for that # noqa
        )
        latest_file = max(list_of_fits, key=os.path.getctime)

        lgbm_laod = load(latest_file)
        feature_set = lgbm_laod.feature_name_
        feature_set.append(self.target)

        return feature_set

    def fit_lgbm(
        self,
        df_train,
        df_val,
        hp_fix_dict,
        hp_tune_dict,
        hp_eval_dict,
        rscv_params,
        feature_set=None,
        reduce_df_mem=True,
        save_model=True,
        learning_rate_decay=True,
        focal_loss=None,  # define fl object here
    ):
        """Run through the entire LGBM modelling pipeline."""
        # Select specific features from df_train for modelling
        if feature_set is not None:
            df_train = df_train[feature_set]
            df_val = df_val[feature_set]

        # Transform categorical columns for lgbm
        df_train = to_categorical(df=df_train)
        df_val = to_categorical(df=df_val)

        # Reduce Memory usage
        if reduce_df_mem:
            df_train = reduce_mem_usage(df=df_train)
            df_val = reduce_mem_usage(df=df_val)

        # Split y and X
        y_train = df_train[self.target]
        X_train = df_train.drop([self.target], axis=1)
        y_val = df_val[self.target]
        X_val = df_val.drop([self.target], axis=1)

        # Insert Validation Set in hp_eval_dict to be used during training
        hp_eval_dict["eval_set"] = [(X_val, y_val)]

        # Define Learning Rate decay and insert into hp_eval_dict
        if learning_rate_decay:

            def learn_rate_decay(current_iter, base=0.1, decay_factor=0.99):
                """Define learning decay shrinkage for better and faster convergence."""
                base_learning_rate = base
                lr = base_learning_rate * np.power(decay_factor, current_iter)
                return lr if lr > 1e-3 else 1e-3

            if "callbacks" in hp_eval_dict:
                hp_eval_dict["callbacks"].append(
                    lgb.reset_parameter(learning_rate=learn_rate_decay)
                )
            else:
                hp_eval_dict["callbacks"] = [
                    lgb.reset_parameter(learning_rate=learn_rate_decay)
                ]

        # Define Focal Loss function and insert into hp_fix_dict and hp_eval_dict
        if focal_loss is not None:
            hp_fix_dict["boost_from_average"] = False
            hp_fix_dict["objective"] = focal_loss.lgb_obj
            hp_eval_dict["eval_metric"] = focal_loss.lgb_eval
            hp_eval_dict["init_score"] = np.full_like(
                y_train, focal_loss.init_score(y_train), dtype=float
            )

        # Call Classifier and HP-Tuner and fit
        lgbm = lgb.LGBMClassifier(boosting_type="gbdt", n_jobs=-1, **hp_fix_dict,)
        lgbm_rscv = RandomizedSearchCV(
            estimator=lgbm, param_distributions=hp_tune_dict, **rscv_params,
        )
        lgbm_fit = lgbm_rscv.fit(X_train, y_train, **hp_eval_dict)

        print(
            "Best score reached: {} with params: {} ".format(
                lgbm_fit.best_score_, lgbm_fit.best_params_
            )
        )

        # Save model
        if save_model:
            time = datetime.now().strftime("%y%m%d%H%M%S")
            dump(
                lgbm_fit.best_estimator_,
                f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/lgbm_fit_{time}.joblib",  # find solution for that # noqa
            )
            if focal_loss is not None:
                dump(
                    focal_loss.init_score(y_train),
                    f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/init_scores/lgbm_fit_{time}.joblib",  # find solution for that # noqa
                )

        return lgbm_fit.best_estimator_

    def predict(
        self, X, predict_from_latest_fit=True, lgbm_fit=None, reduce_df_mem=True
    ):
        """Predict for X Churn probabibilities."""
        # Load model or use passed fit
        if lgbm_fit is not None and not predict_from_latest_fit:
            lgbm = lgbm_fit
        elif predict_from_latest_fit and lgbm_fit is None:
            list_of_fits = glob.glob(
                "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/*"  # find solution for that # noqa
            )
            latest_file = max(list_of_fits, key=os.path.getctime)
            last_part = latest_file.rsplit("lgbm_fit_", 1)[1]
            lgbm = load(latest_file)
        else:
            raise ValueError(
                "Either define only lgbm_fit or set predict_from_latest_fit to True"
            )  # noqa

        # Use same features used for fitting loaded model
        feature_set = lgbm.feature_name_
        X = X[feature_set]

        # Small preprocessing
        X = to_categorical(df=X)
        if reduce_df_mem:
            X = reduce_mem_usage(X)

        # Check wether custom loss was used for correct probability estimates
        custom_objective = lgbm.get_params()["objective"] != "binary"
        if custom_objective:  # expected cached init_score to add to preds
            init_score = load(
                f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/init_scores/lgbm_fit_{last_part}"  # find solution for that # noqa
            )
            preds_proba = expit(init_score + lgbm.predict(X))
            preds = (preds_proba >= 0.5).astype("int")
        else:
            preds = lgbm.predict(X)
            preds_proba2 = lgbm.predict_proba(X)
            preds_proba = [p[1] for p in preds_proba2]

        return preds, preds_proba

    # def explain(
    #     self, df, explain_from_latest_fit=True, lgbm_fit=None, reduce_df_mem=True
    # ):