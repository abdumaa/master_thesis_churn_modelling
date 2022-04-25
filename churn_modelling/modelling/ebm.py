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

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.mrmr import mrmr
from churn_modelling.utils.mem_usage import reduce_mem_usage

empty_df = pd.DataFrame()


class EBM:
    """Class for ebm modelling and predictions for probabilities of midcontract churns."""

    def __init__(self, df, target, test_size=0.1):
        self.df = df
        self.target = target
        self.test_size = test_size

    def create_train_test(self):
        """Split train and test."""
        train, test = split_train_test(
            df=self.df, target=self.target, test_size=self.test_size
        )
        return train, test

    def create_sampling(self, df_to_sample=None, sampling="down", frac="balanced"):
        """Create (synthetic) up- or down-sampling."""
        if df_to_sample is not None:
            return resample(
                df_to_sample=df_to_sample, target=self.target, sampling=sampling, frac=frac
            )
        else:
            return resample(
                df_to_sample=self.df, target=self.target, sampling=sampling, frac=frac
            )

    def split_quotation_fix_features(self):
        """Return two lists containing quotation & fix features."""
        full_features = self.df.columns.to_list()
        full_features.remove(self.target)
        quotation_features = [i for i in full_features if "requests" in i]
        fix_features = [i for i in full_features if "requests" not in i]
        return quotation_features, fix_features

    def feature_selection(self, df_to_dimreduce=None, variable_names=None, cv=5, sample=False):
        """Create feature_selection_df using MRMR-Scorer."""
        if df_to_dimreduce is not None:
            if variable_names is not None:
                variable_names.append(self.target)
                df_to_dimreduce = df_to_dimreduce[variable_names]
            return mrmr(
                df_to_dimreduce=df_to_dimreduce,
                target=self.target,
                cv=cv,
                sample=sample,
            )
        else:
            if variable_names is not None:
                variable_names.append(self.target)
                df_to_dimreduce = self.df[variable_names]
            return mrmr(
                df_to_dimreduce=df_to_dimreduce, target=self.target, cv=cv, sample=sample
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

    def get_best_quot_features(
        self,
        df_to_dimreduce=None,
        cv=5,
        sample=False,
        include_fix_features=True,
        include_target=True,
    ):
        # Split quotation and fix features
        quot_features, fix_features = self.split_quotation_fix_features()

        # Perform MRMR
        iterations_df = self.feature_selection(
            df_to_dimreduce=df_to_dimreduce,
            variable_names=quot_features,
            cv=cv,
            sample=sample,
        )

        # Get best set of quotation features
        feature_order = ast.literal_eval(iterations_df["SELECTED_SET"].iloc[-1])
        quot_feats_groups = []
        best_feats = []
        for i in feature_order:
            if i[:-2] not in quot_feats_groups:
                quot_feats_groups.append(i[:-2])
                best_feats.append(i)

        # Append fix features and target if set to true
        if include_fix_features:
            best_feats.extend(fix_features)
        if include_target:
            best_feats.append(self.target)

        return best_feats

    def get_featureset_from_fit(self, fit, include_target=True, include_interactions=False):
        """Extract feature set based on specific fit."""

        features_used = fit.feature_names
        single_features = []
        interaction_features = []
        for f in features_used:
            if ' x ' in f:
                interaction_features.append(f)
            else:
                single_features.append(f)

        if include_target:
            single_features.append(self.target)

        if include_interactions:
            return single_features, interaction_features
        else:
            return single_features

    def fit_ebm(
        self,
        df_train,
        hp_fix_dict,
        hp_tune_dict,
        rscv_params,
        seed,
        feature_set=None,
        reduce_df_mem=True,
        save_model=True,
        cache_model_name=None,
    ):
        """Run through the entire EBM modelling pipeline."""
        # Select specific features from df_train for modelling
        print("..1: Small preprocessing")
        if feature_set is not None:
            df_train = df_train[feature_set]
            df_val = df_val[feature_set]

        # Transform categorical columns for ebm
        df_train = to_categorical(df=df_train)

        # Reduce Memory usage
        if reduce_df_mem:
            df_train = reduce_mem_usage(df=df_train)

        # Split y and X
        y_train = df_train[self.target]
        X_train = df_train.drop([self.target], axis=1)

        # Call Classifier and HP-Tuner and fit
        print("..2: Start CV-HP-Tuning")
        ebm = ExplainableBoostingClassifier(n_jobs=-1, random_state=seed, **hp_fix_dict)
        ebm_rscv = RandomizedSearchCV(
            estimator=ebm, param_distributions=hp_tune_dict, **rscv_params,
        )
        ebm_fit = ebm_rscv.fit(X_train, y_train)

        print("..3: Finished CV-HP-Tuning")
        # print(
        #     "Best score reached: {} with params: {} ".format(
        #         ebm_fit.best_score_, ebm_fit.best_params_
        #     )
        # )

        # Save model
        if save_model:
            print("..4: Save best model")
            dump(
                ebm_fit.best_estimator_,
                f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/ebm_fit_{cache_model_name}.joblib",  # find solution for that # noqa
            )

        return ebm_fit.best_estimator_

    def predict(
        self, X, predict_from_cached_fit=True, fit=None, cache_model_name=None, reduce_df_mem=True
    ):
        """Predict for X Churn probabibilities."""
        # Load model or use passed fit
        if fit is not None and not predict_from_cached_fit:
            ebm = fit
        elif predict_from_cached_fit and fit is None:
            ebm = load(
                f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/ebm_fit_{cache_model_name}.joblib"  # find solution for that # noqa
            )
        else:
            raise ValueError(
                "Either define only ebm_fit or set predict_from_cached_fit to True"
            )  # noqa

        # Use same features used for fitting loaded model
        feature_set = self.get_featureset_from_fit(ebm, include_target=False)
        X = X[feature_set]

        # Small preprocessing
        X = to_categorical(df=X)
        if reduce_df_mem:
            X = reduce_mem_usage(X)

        # Predict
        preds = ebm.predict(X)
        preds_proba2 = ebm.predict_proba(X)
        preds_proba = [p[1] for p in preds_proba2]

        return preds, preds_proba