import pandas as pd

import numpy as np
import ast
import glob
import os
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)

import seaborn as sns
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.mrmr import mrmr
from churn_modelling.utils.mem_usage import reduce_mem_usage
from churn_modelling.modelling.utils import split_quotation_fix_features, get_featureset_from_fit

empty_df = pd.DataFrame()


class EBM:
    """Class for ebm modelling and predictions for probabilities of midcontract churns."""

    def __init__(self, df_trainval=None, df_oos=None, df_oop=None, target="churn", test_size=0.1):
        """Initialize class attributes.

        Args:
            df_trainval (pandas.DataFrame, optional):
                Data set containing training and validation set. Defaults to None.
            df_oos (pandas.DataFrame, optional):
                Data set containing out of sample test set. Defaults to None.
            df_oop (pandas.DataFrame, optional):
                Data set containing out of period test set. Defaults to None.
            target (str, optional):
                Name of the target variable in the data sets. Defaults to "churn".
            test_size (float, optional):
                size of the out of sample test set. Defaults to 0.1.
        """

        if df_trainval is not None:
            self.df = df_trainval
        else:
            self.df = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
        if df_oos is not None:
            self.df_oos = df_oos
        else:
            self.df_oos = pd.read_csv('../data/toydata_oos.csv', index_col=0)
        if df_oop is not None:
            self.df_oop = df_oop
        else:
            self.df_oop = pd.read_csv('../data/toydata_oop.csv', index_col=0)
        self.target = target
        self.test_size = test_size

    def create_sampling(self, df_to_sample=None, sampling="down", frac="balanced"):
        """Create (synthetic) up- or down-sampling.

        Args:
            df_to_sample (pandas.DataFrame, optional):
                Data set to create sampling on. Defaults to None.
            sampling (str, optional):
                - Sampling method.
                - Either "down", "up" or "smote".
                - Defaults to "down".
            frac (str or float, optional):
                - Defines sampling fraction.
                - Either "balanced" or float.
                - For "balanced" samples such that both classes have same length.
                - For float multiplies min/maj class with float.
                - Defaults to "balanced".

        Returns:
            pandas.DataFrame: Sampled data set.
        """
        if df_to_sample is not None:
            return resample(
                df_to_sample=df_to_sample, target=self.target, sampling=sampling, frac=frac
            )
        else:
            return resample(
                df_to_sample=self.df, target=self.target, sampling=sampling, frac=frac
            )

    def get_best_quot_features(
        self,
        df_to_dimreduce=None,
        cv=5,
        sample=False,
        return_fix_features=True,
        return_target=True,
    ):
        """Get the best set of quotation features using MRMR.

        Args:
            df_to_dimreduce (pandas.DataFrame, optional):
                Data set to create dimension reduction on. Defaults to None.
            cv (int, optional):
                Number of folds to estimate relevance scores in MRMR. Defaults to 5.
            return_fix_features (bool, optional):
                Wether to return the fix features also or not. Defaults to True.
            return_target (bool, optional):
                Wether to return the target variable also or not. Defaults to True.

        Returns:
            pandas.DataFrame: Dimension reduced data set
        """

        # Split quotation and fix features
        quot_features, fix_features = split_quotation_fix_features(df_to_dimreduce)

        # Perform MRMR
        iterations_df = mrmr(
            df_to_dimreduce=df_to_dimreduce,
            variable_names=quot_features,
            cv=cv,
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
        if return_fix_features:
            best_feats.extend(fix_features)
        if return_target:
            best_feats.append(self.target)

        return best_feats

    def fit_ebm(
        self,
        df_train,
        hp_fix_dict,
        hp_tune_dict,
        rscv_params,
        feature_set=None,
        reduce_df_mem=True,
        save_model=True,
        cache_model_name="test",
        path_to_folder=None,
    ):
        """Run through the entire EBM M-HPTL loop pipeline.

        Args:
            df_train (pandas.DataFrame):
                Training data set
            hp_fix_dict (dict):
                Dictionary containing fix parameters for ebm
            hp_tune_dict (dict):
                Dictionary containing to be tuned parameters for ebm
            rscv_params (dict):
                Dictionary containing tuning process parameters for sklearn.rscv
            feature_set (list, optional):
                Features to be used during training. Defaults to None.
            reduce_df_mem (bool, optional):
                Wether to reduce mem usage of data sets. Defaults to True.
            save_model (bool, optional):
                Wether to save model as "cache_model_name" in "path_to_folder/ebm_fits/". Defaults to True.
            cache_model_name (str, optional):
                Name of saved model. Defaults to "test".
            path_to_folder (str, optional):
                - Path to folder which contains folders "ebm_fits/"
                and where model should be saved.
                - Defaults to None.
                - If None takes file path of executed file

        Returns:
            ebm model: Returns the best fit of the M-HPTL
        """
        # Select specific features from df_train for modelling
        print("..1: Small preprocessing")
        if feature_set is not None:
            df_train = df_train[feature_set]

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
        ebm = ExplainableBoostingClassifier(n_jobs=-1, **hp_fix_dict)
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
            if path_to_folder is None:
                path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
            dump(
                ebm_fit.best_estimator_,
                f"{path_to_folder}/ebm_fits/{cache_model_name}.joblib",  # find solution for that # noqa
            )

        return ebm_fit.best_estimator_

    def predict(
        self,
        X,
        predict_from_cached_fit=True,
        fit=None,
        cache_model_name=None,
        path_to_folder=None,
        reduce_df_mem=True
    ):
        """Predict churn probabibilities for X.

        Args:
            X (pandas.DataFrame):
                Data set to predict probabilities for
            predict_from_cached_fit (bool, optional):
                Wether to predict from cached fit. Defaults to True.
            fit (ebm model, optional):
                Model to use for predictions. Defaults to None.
            cache_model_name (str, optional):
                Name of cached model to use for predictions. Defaults to "test".
            path_to_folder ([type], optional):
                - Path to folder which contains folders "ebm_fits/"
                and where models are saved.
                - Defaults to None.
                - If None takes file path of executed file
            reduce_df_mem (bool, optional):
                Wether to reduce mem usage of X. Defaults to True.

        Raises:
            ValueError: If fit and predict_from_cached_fit are defined or not not defined

        Returns:
            tuple: (prediction labels, predicted probabilities)
        """

        # Load model or use passed fit
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
        if fit is not None and not predict_from_cached_fit:
            ebm = fit
        elif predict_from_cached_fit and fit is None:
            ebm = load(
                f"{path_to_folder}/ebm_fits/{cache_model_name}.joblib"  # find solution for that # noqa
            )
        else:
            raise ValueError(
                "Either define only ebm_fit or set predict_from_cached_fit to True"
            )  # noqa

        # Use same features used for fitting loaded model
        feature_set = get_featureset_from_fit(ebm, include_target=False)
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

    def fit_and_eval_ebm_candidates(
        self,
        hp_struct_dict,
        hp_fix_dict,
        hp_tune_dict,
        rscv_params,
        reduce_df_mem=True,
        path_to_folder=None,
        feature_set_from_last_fits=True,
    ):
        """Run through the entire EBM S-HPTL loop pipeline.

        Args:
            hp_struct_dict (dict):
                Dictionary containing parameters for S-HPTL
                Must contain: sampling and dr_method
            hp_fix_dict (dict):
                Dictionary containing fix parameters for ebm
            hp_tune_dict (dict):
                Dictionary containing to be tuned parameters for ebm
            rscv_params (dict):
                Dictionary containing tuning process parameters for sklearn.rscv
            reduce_df_mem (bool, optional):
                Wether to reduce mem usage of data sets. Defaults to True.
            path_to_folder (str, optional):
                - Path to folder which contains folders "ebm_fits/" and "init_scores/"
                and where model should be saved.
                - Defaults to None.
                - If None takes file path of executed file
            feature_set_from_last_fits (bool, optional):
                - Wether to look for a fit with same sampling method and take features from it.
                - Defaults to True.

        Returns:
            pandas.DataFrame: Returns the summary dataframe
        """

        # Define path_to_folder
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script

        # Define df_train
        df_train = self.df

        # Sampling
        cache_dict = {}
        for sampling_method in hp_struct_dict['sampling'].keys():
            if sampling_method == 'none':
                if hp_struct_dict['sampling']['none']:
                    df_train_sampled = df_train
            if sampling_method == 'smote':
                if hp_struct_dict['sampling']['smote']:
                    df_train_sampled = self.create_sampling(df_to_sample=df_train, sampling="smote")
            if sampling_method[:-1] == 'down':
                df_train_sampled = self.create_sampling(
                    df_to_sample=df_train,
                    sampling="down",
                    frac=hp_struct_dict['sampling'][sampling_method]
                )
            if sampling_method[:-1] == 'up':
                df_train_sampled = self.create_sampling(
                    df_to_sample=df_train,
                    sampling="up",
                    frac=hp_struct_dict['sampling'][sampling_method]
                )

            # Dimension Reduction
            if feature_set_from_last_fits:
                list_of_fits = glob.glob(
                    "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/ modelling/ebm_fits/*"  # find solution for that # noqa
                )
                list_of_fits_filtered = [x for x in list_of_fits if sampling_method in x.rsplit("ebm_fit_", 1)[1]]
                latest_file = max(list_of_fits_filtered, key=os.path.getctime)
                ebm_laod = load(latest_file)
                best_feats = get_featureset_from_fit(ebm_laod)
            else:
                best_feats = self.get_best_quot_features(
                    df_to_dimreduce=df_train_sampled
                )
            quot_feats, fix_features = split_quotation_fix_features(df_train_sampled)

            for dr_method in hp_struct_dict['dr_method']:
                if dr_method == 'no_quot':
                    fix_features.append("churn")
                    features_used = fix_features
                if dr_method == 'best_quot':
                    features_used = best_feats
                df_train_dr = df_train_sampled[features_used]
                features_used.remove("churn")

                ### Fit models ###
                # EBM hyperparameter dictionaries
                cache_model_name = f"ebm_fit_ebm_{sampling_method}_{dr_method}"
                fix_dict = hp_fix_dict
                tune_dict = hp_tune_dict
                cv_params = rscv_params
                best_fit = self.fit_ebm(
                    df_train=df_train_dr,
                    hp_fix_dict=fix_dict,
                    hp_tune_dict=tune_dict,
                    rscv_params=cv_params,
                    reduce_df_mem=True,
                    save_model=True,
                    cache_model_name=cache_model_name,
                    path_to_folder=path_to_folder,
                )

                # Predict oos and oop
                preds_oos, preds_proba_oos = self.predict(
                    self.df_oos[features_used],
                    predict_from_cached_fit=False,
                    fit=best_fit,
                    cache_model_name=cache_model_name,
                )
                preds_oop, preds_proba_oop = self.predict(
                    self.df_oop[features_used],
                    predict_from_cached_fit=False,
                    fit=best_fit,
                    cache_model_name=cache_model_name,
                )

                # Evaluate results
                acc_oos = accuracy_score(self.df_oos["churn"], preds_oos)
                prec_oos = precision_score(self.df_oos["churn"], preds_oos)
                rec_oos = recall_score(self.df_oos["churn"], preds_oos)
                f1_oos = f1_score(self.df_oos["churn"], preds_oos)
                auroc_oos = roc_auc_score(self.df_oos["churn"], preds_proba_oos)
                auprc_oos = average_precision_score(self.df_oos["churn"], preds_proba_oos)

                acc_oop = accuracy_score(self.df_oop["churn"], preds_oop)
                prec_oop = precision_score(self.df_oop["churn"], preds_oop)
                rec_oop = recall_score(self.df_oop["churn"], preds_oop)
                f1_oop = f1_score(self.df_oop["churn"], preds_oop)
                auroc_oop = roc_auc_score(self.df_oop["churn"], preds_proba_oop)
                auprc_oop = average_precision_score(self.df_oop["churn"], preds_proba_oop)

                # Cache results
                cache_dict[cache_model_name] = {
                    'sampling': f"{sampling_method}: {hp_struct_dict['sampling'][sampling_method]}",
                    'dr_method': f"{dr_method}",
                    'features_after_dr': features_used,
                    # 'interactions_after_FAST': interaction_feats,
                    'model': best_fit,
                    'Accuracy_OOS': acc_oos,
                    'Precision_OOS': prec_oos,
                    'Recall_OOS': rec_oos,
                    'F1_Score_OOS': f1_oos,
                    'AUROC_OOS': auroc_oos,
                    'AUPRC_OOS': auprc_oos,
                    'Accuracy_OOP': acc_oop,
                    'Precision_OOP': prec_oop,
                    'Recall_OOP': rec_oop,
                    'F1_Score_OOP': f1_oop,
                    'AUROC_OOP': auroc_oop,
                    'AUPRC_OOP': auprc_oop,
                    'path': f"{path_to_folder}/ebm_fits/{cache_model_name}.joblib"
                }
        cache_df = pd.DataFrame.from_dict(cache_dict, orient='index').sort_values(by=['AUPRC_OOP', 'AUPRC_OOS'],    ascending=False)
        cache_df.to_csv(f'ebm_results/results.csv')
        return cache_df

    def plot_shape_function(self, feats, cache_model_name, path_to_folder=None, save=True):
        """Plot shape function(s) k() for a given EBM model and feature(s).

        Args:
            feats (list):
                list of features to plot shape functions for.
            cache_model_name (str):
                Name of saved model.
            path_to_folder (str, optional):
                - Path to folder which contains folders "ebm_fits/" and "plots/"
                and where plots should be saved.
                - Defaults to None.
                - If None takes file path of executed file
            save (bool, optional):
                Saves plots in "path_to_folder/plots/". Defaults to True.
        """

        # Load model
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
        path = f"{path_to_folder}/ebm_fits/{cache_model_name}.joblib"
        model_ebm = load(path)

        # Load EBM interpreter
        ebm_global = model_ebm.explain_global()

        # Create plots for features
        for f in feats:
            print(f)
            # Extract data from fitted shape function
            idx = ebm_global.selector.index[ebm_global.selector['Name'] == f][0]
            data_dict = ebm_global.data(idx)
            x_vals = data_dict["names"].copy()
            y_vals = data_dict["scores"].copy()

            # Rearrangements
            y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]
            x = np.array(x_vals)

            # Create plot
            sns.lineplot(x, y_vals, drawstyle='steps-post')
            plt.xlabel(f)
            plt.ylabel(f'$k_{idx+1}$({f})')
            plt.show()
            if save:
                plt.savefig(f"{path_to_folder}/plots/shape_function_{f}.png", dpi=100, bbox_inches='tight')
            plt.clf()