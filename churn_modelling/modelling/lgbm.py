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

from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
import lightgbm as lgb

from churn_modelling.preprocessing.categorical import to_categorical
from churn_modelling.preprocessing.splitting import split_train_test
from churn_modelling.preprocessing.resample import resample
from churn_modelling.preprocessing.mrmr import mrmr
from churn_modelling.utils.mem_usage import reduce_mem_usage
from .utils import split_quotation_fix_features, get_featureset_from_fit, get_featureset_from_cached_fit
from .custom_loss import FocalLoss, WeightedLoss


empty_df = pd.DataFrame()

class LGBM:
    """Class for lgbm preprocessing, modelling, predicting, evaluating and interpreting."""

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

    def create_train_val(self):
        """Split into train and val set.

        Returns:
            tuple: Tuple of (train, val) as pandas.DataFrames.
        """

        train, val = split_train_test(
            df=self.df,
            target=self.target,
            test_size=self.test_size / (1 - self.test_size),
        )
        return train, val

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
        learning_rate_decay=True,
        cl_alpha=None,
        cl_gamma=None,
        save_model=True,
        cache_model_name="test",
        path_to_folder=None,
    ):
        """Run through the entire GBT M-HPTL loop pipeline.

        Args:
            df_train (pandas.DataFrame):
                Training data set
            df_val (pandas.DataFrame):
                Validation data set
            hp_fix_dict (dict):
                Dictionary containing fix parameters for lgbm
            hp_tune_dict (dict):
                Dictionary containing to be tuned parameters for lgbm
            hp_eval_dict (dict):
                Dictionary containing evaluation parameters for lgbm
            rscv_params (dict):
                Dictionary containing tuning process parameters for sklearn.rscv
            feature_set (list, optional):
                Features to be used during training. Defaults to None.
            reduce_df_mem (bool, optional):
                Wether to reduce mem usage of data sets. Defaults to True.
            learning_rate_decay (bool, optional): 
                Wether to use decaying learning rate. Defaults to True.
            cl_alpha (float, optional):
                Alpha weight for custom loss. Defaults to None.
            cl_gamma (float, optional):
                Gamma parameter for focal loss. Defaults to None.
            save_model (bool, optional):
                Wether to save model as "cache_model_name" in "path_to_folder/lgbm_fits/". Defaults to True.
            cache_model_name (str, optional):
                Name of saved model. Defaults to "test".
            path_to_folder (str, optional):
                - Path to folder which contains folders "lgbm_fits/" and "init_scores/"
                and where model should be saved.
                - Defaults to None.
                - If None takes file path of executed file

        Returns:
            lgbm model: Returns the best fit of the M-HPTL
        """

        print("..1: Small preprocessing")
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

        # Define Custom Loss function and insert into hp_fix_dict and hp_eval_dict
        if cl_alpha is None and cl_gamma is None:
            custom_loss = None
        elif cl_gamma is None and cl_alpha is not None:
            custom_loss = None
            hp_fix_dict["scale_pos_weight"] = (cl_alpha / (1 - cl_alpha)) # same as: custom_loss = WeightedLos(alpha=cl_alpha)
        else:
            custom_loss = FocalLoss(alpha=cl_alpha, gamma=cl_gamma)
            hp_fix_dict["boost_from_average"] = False
            hp_fix_dict["objective"] = custom_loss.lgb_obj
            hp_eval_dict["eval_metric"] = custom_loss.lgb_eval
            hp_eval_dict["init_score"] = np.full_like(
                y_train, custom_loss.init_score(y_train), dtype=float
            )

        # Call Classifier and HP-Tuner and do HP-Tuning
        print("..2: Start CV-HP-Tuning")
        lgbm = lgb.LGBMClassifier(boosting_type="gbdt", n_jobs=-1, **hp_fix_dict,)
        lgbm_rscv = RandomizedSearchCV(
            estimator=lgbm, param_distributions=hp_tune_dict, **rscv_params,
        )
        lgbm_fit = lgbm_rscv.fit(X_train, y_train, **hp_eval_dict)
        print("..3: Finished CV-HP-Tuning")
        print(
            "Best score reached: {} with params: {} ".format(
                lgbm_fit.best_score_, lgbm_fit.best_params_
            )
        )

        # Save model
        if save_model:
            print("..4: Save best model")
            if path_to_folder is None:
                path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
            dump(
                lgbm_fit.best_estimator_,
                f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib",
            )
            if custom_loss is not None:
                dump(
                    custom_loss.init_score(y_train),
                    f"{path_to_folder}/init_scores/{cache_model_name}.joblib",
                )

        return lgbm_fit.best_estimator_

    def predict(
        self,
        X,
        predict_from_cached_fit=True,
        fit=None,
        cache_model_name="test",
        path_to_folder=None,
        reduce_df_mem=True
    ):
        """Predict churn probabibilities for X.

        Args:
            X (pandas.DataFrame):
                Data set to predict probabilities for
            predict_from_cached_fit (bool, optional):
                Wether to predict from cached fit. Defaults to True.
            fit (lgbm model, optional):
                Model to use for predictions. Defaults to None.
            cache_model_name (str, optional):
                Name of cached model to use for predictions. Defaults to "test".
            path_to_folder ([type], optional):
                - Path to folder which contains folders "lgbm_fits/" and "init_scores/"
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
            lgbm = fit
        elif predict_from_cached_fit and fit is None:
            lgbm = load(
                f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib"
            )
        else:
            raise ValueError(
                "Either define only lgbm_fit or set predict_from_cached_fit to True"
            ) # noqa

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
                f"{path_to_folder}/init_scores/{cache_model_name}.joblib"
            )
            preds_proba = expit(init_score + lgbm.predict(X))
            preds = (preds_proba >= 0.5).astype("int")
        else:
            preds = lgbm.predict(X)
            preds_proba2 = lgbm.predict_proba(X)
            preds_proba = [p[1] for p in preds_proba2]

        return preds, preds_proba

    def fit_and_eval_lgbm_candidates(
        self,
        hp_struct_dict,
        hp_fix_dict,
        hp_tune_dict,
        hp_eval_dict,
        rscv_params,
        reduce_df_mem=True,
        learning_rate_decay=True,
        path_to_folder=None,
        feature_set_from_last_fits=True,
    ):
        """Run through the entire GBT S-HPTL loop pipeline.

        Args:
            hp_struct_dict (dict):
                Dictionary containing parameters for S-HPTL
                Must contain: cl_alpha, cl_gamma, sampling and dr_method
            hp_fix_dict (dict):
                Dictionary containing fix parameters for lgbm
            hp_tune_dict (dict):
                Dictionary containing to be tuned parameters for lgbm
            hp_eval_dict (dict):
                Dictionary containing evaluation parameters for lgbm
            rscv_params (dict):
                Dictionary containing tuning process parameters for sklearn.rscv
            reduce_df_mem (bool, optional):
                Wether to reduce mem usage of data sets. Defaults to True.
            learning_rate_decay (bool, optional):
                Wether to use decaying learning rate. Defaults to True.
            path_to_folder (str, optional):
                - Path to folder which contains folders "lgbm_fits/" and "init_scores/"
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

        # Split df into train and val
        df_train, df_val = self.create_train_val()

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
                    f"{path_to_folder}/lgbm_fits/*"
                )
                list_of_fits_filtered = [x for x in list_of_fits if sampling_method in x.rsplit("lgbm_fit_", 1)[1]]
                latest_file = max(list_of_fits_filtered, key=os.path.getctime)
                lgbm_laod = load(latest_file)
                best_feats = lgbm_laod.feature_name_
                best_feats.append("churn")
            else:
                best_feats = self.get_best_quot_features(
                    df_to_dimreduce=df_train_sampled
                )
            quot_feats, fix_features = split_quotation_fix_features(df=df_train_sampled)

            for dr_method in hp_struct_dict['dr_method']:
                if dr_method == 'no_quot':
                    fix_features.append("churn")
                    df_train_dr = df_train_sampled[fix_features]
                    df_val_dr = df_val[fix_features]
                if dr_method == 'best_quot':
                    df_train_dr = df_train_sampled[best_feats]
                    df_val_dr = df_val[best_feats]

            # Define objective loss function
                for alpha in hp_struct_dict['cl_alpha']:
                    for gamma in hp_struct_dict['cl_gamma']:
                        ### Fit models ###
                        # LGBM hyperparameter dictionaries
                        cache_model_name = f"lgbm_fit_gbt_{sampling_method}_{dr_method}_a{alpha}_g{gamma}"
                        fix_dict = hp_fix_dict
                        tune_dict = hp_tune_dict
                        eval_dict = hp_eval_dict
                        cv_params = rscv_params
                        best_fit = self.fit_lgbm(
                            df_train=df_train_dr,
                            df_val=df_val_dr,
                            hp_fix_dict=fix_dict,
                            hp_tune_dict=tune_dict,
                            hp_eval_dict=eval_dict,
                            rscv_params=cv_params,
                            reduce_df_mem=True,
                            save_model=True,
                            learning_rate_decay=learning_rate_decay,
                            cl_alpha=alpha,
                            cl_gamma=gamma,
                            cache_model_name=cache_model_name,
                            path_to_folder=path_to_folder,
                        )

                        # Predict oos and oop
                        preds_oos, preds_proba_oos = self.predict(
                            self.df_oos,
                            predict_from_cached_fit=False,
                            fit=best_fit,
                            cache_model_name=cache_model_name,
                        )
                        preds_oop, preds_proba_oop = self.predict(
                            self.df_oop,
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
                            'features_after_dr': list(df_train_dr.columns),
                            'loss': {'alpha': alpha, 'gamma': gamma},
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
                            'path': f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib"
                        }
        cache_df = pd.DataFrame.from_dict(cache_dict, orient='index').sort_values(by=['F1_Score_OOP', 'F1_Score_OOS'],
            ascending=False)
        cache_df.to_csv(f'lgbm_results/results.csv')
        return cache_df

    def plot_pdp(self, feats, cache_model_name, path_to_folder=None, save=True):
        """Plot PDP for a given model and features.

        Args:
            feats (list):
                list of features to plot PDPs for.
            cache_model_name (str):
                Name of saved model.
            path_to_folder (str, optional):
                - Path to folder which contains folders "lgbm_fits/" and "plots/"
                and where plots should be saved.
                - Defaults to None.
                - If None takes file path of executed file
            save (bool, optional):
                Saves plots in "path_to_folder/plots/". Defaults to True.
        """

        # Load model
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
        path = f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib"
        model_gbt = load(path)

        # Split y and X
        features = model_gbt.feature_name_
        X = self.df_oos[features]

        # Create PDP-Plots
        for f in feats:
            PartialDependenceDisplay.from_estimator(model_gbt, X, [f], target=self.target, n_jobs=-1)
            if save:
                plt.savefig(f"{path_to_folder}/plots/pdp_{f}.png", dpi=100, bbox_inches='tight')

    def plot_shap_summary(self, cache_model_name, path_to_folder=None, save=True):
        """Plot SHAP summary for a given model.

        Args:
            cache_model_name (str):
                Name of saved model.
            path_to_folder (str, optional):
                - Path to folder which contains folders "lgbm_fits/" and "plots/"
                and where plots should be saved.
                - Defaults to None.
                - If None takes file path of executed file
            save (bool, optional):
                Saves plots in "path_to_folder/plots/". Defaults to True.

        Returns:
            fig: Figure of summary plot
        """

        # Load model
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
        path = f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib"
        model_gbt = load(path)

        # Split y and X
        features = model_gbt.feature_name_
        X = self.df_oos[features]
        y = self.df_oos[self.target]

        # Calculate shap values
        explainer = shap.TreeExplainer(model_gbt)
        shap_values = explainer.shap_values(X)

        # Plot summary plot
        fig = shap.summary_plot(shap_values[1], X, color_bar_label='Variable value', show=False)

        if save:
            plt.savefig(f"{path_to_folder}/plots/shap_summary.png", dpi=200, bbox_inches='tight')

        return fig

    def plot_shap_waterfall(self, idx, cache_model_name, path_to_folder=None, save=True):
        """Plot SHAP's waterfall plot for a given model and a sample.

        Args:
            idx (int):
                index of sample to be explained.
            cache_model_name (str):
                Name of saved model.
            path_to_folder (str, optional):
                - Path to folder which contains folders "lgbm_fits/" and "plots/"
                and where plots should be saved.
                - Defaults to None.
                - If None takes file path of executed file
            save (bool, optional):
                Saves plots in "path_to_folder/plots/". Defaults to True.
        """

        # Load model
        if path_to_folder is None:
            path_to_folder = "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling" # Maybe define function to get current path of the executing script
        path = f"{path_to_folder}/lgbm_fits/{cache_model_name}.joblib"
        model_gbt = load(path)

        # Split y and X
        features = model_gbt.feature_name_
        X = self.df_oos[features]
        y = self.df_oos[self.target]

        # Calculate shap values
        explainer = shap.Explainer(model_gbt, X)
        shap_values = explainer(X.loc[[idx]])

        # Plot summary plot
        shap.plots.waterfall(shap_values[0], show=True)

        if save:
            plt.gcf()
            plt.savefig(f"{path_to_folder}/plots/shap_waterfall_{idx}.png", dpi=200, bbox_inches='tight')
