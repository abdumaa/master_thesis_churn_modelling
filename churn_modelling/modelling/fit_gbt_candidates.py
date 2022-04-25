import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import os
import glob
from joblib import load
from lgbm import LGBM
from custom_loss import FocalLoss, WeightedLoss

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform



def create_fits_and_predictions(sampling_dict, cl_alpha=None, cl_gamma=None, seed=123, feature_set_from_last_fits=False):
    """Create Fits for different sampling methods with LGBMClassifier."""

    # Load raw flatfiles
    df = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
    df_oos = pd.read_csv('../data/toydata_oos.csv', index_col=0)
    df_oop = pd.read_csv('../data/toydata_oop.csv', index_col=0)

    # Call model and split df into train and val
    model_pl = LGBM(df=df, target="churn", test_size=0.1)
    df_train, df_val = model_pl.create_train_val()

    # Sampling
    cache_dict = {}
    for sampling_method in sampling_dict['sampling'].keys():
        if sampling_method == 'none':
            if sampling_dict['sampling']['none']:
                df_train_sampled = df_train
        if sampling_method == 'smote':
            if sampling_dict['sampling']['smote']:
                df_train_sampled = model_pl.create_sampling(df_to_sample=df_train, sampling="smote")
        if sampling_method[:-1] == 'down':
            df_train_sampled = model_pl.create_sampling(
                df_to_sample=df_train,
                sampling="down",
                frac=sampling_dict['sampling'][sampling_method]
            )
        if sampling_method[:-1] == 'up':
            df_train_sampled = model_pl.create_sampling(
                df_to_sample=df_train,
                sampling="up",
                frac=sampling_dict['sampling'][sampling_method]
            )

        # Dimension Reduction
        if feature_set_from_last_fits:
            list_of_fits = glob.glob(
                "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/*"  # find solution for that # noqa
            )
            list_of_fits_filtered = [x for x in list_of_fits if sampling_method in x.rsplit("lgbm_fit_", 1)[1]]
            latest_file = max(list_of_fits_filtered, key=os.path.getctime)
            lgbm_laod = load(latest_file)
            best_feats = lgbm_laod.feature_name_
            best_feats.append("churn")
        else:
            best_feats = model_pl.get_best_quot_features(
                df_to_dimreduce=df_train_sampled
            )
        quot_feats, fix_features = model_pl.split_quotation_fix_features()

        for dr_method in sampling_dict['dr_method']:
            if dr_method == 'no_quot':
                fix_features.append("churn")
                df_train_dr = df_train_sampled[fix_features]
                df_val_dr = df_val[fix_features]
            if dr_method == 'best_quot':
                df_train_dr = df_train_sampled[best_feats]
                df_val_dr = df_val[best_feats]

            ### Fit models ###
            # LGBM hyperparameter dictionaries
            hp_fix_dict = {
                "objective": "binary",
                "max_depth": -1,
                "n_estimators": 1000,
                "random_state": seed,
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
                "random_state": seed,
                "n_jobs": -1,
                "cv": 3,
                "verbose": 100,
            }
            if cl_alpha is None and cl_gamma is None:
                custom_loss = None
            elif cl_gamma is None and cl_alpha is not None:
                custom_loss = None
                hp_fix_dict["scale_pos_weight"] = (cl_alpha / (1 - cl_alpha)) # same as: custom_loss = WeightedLoss(alpha=cl_alpha)
            else:
                custom_loss = FocalLoss(alpha=cl_alpha, gamma=cl_gamma)
            cache_model_name = f"gbt_{sampling_method}_{dr_method}_a{cl_alpha}_g{cl_gamma}"
            best_fit = model_pl.fit_lgbm(
                df_train=df_train_dr,
                df_val=df_val_dr,
                hp_fix_dict=hp_fix_dict,
                hp_tune_dict=hp_tune_dict,
                hp_eval_dict=hp_eval_dict,
                rscv_params=rscv_params,
                reduce_df_mem=True,
                save_model=True,
                learning_rate_decay=True,
                custom_loss=custom_loss,
                cache_model_name=cache_model_name,
            )

            # Predict oos and oop
            preds_oos, preds_proba_oos = model_pl.predict(
                df_oos,
                predict_from_cached_fit=False,
                fit=best_fit,
                cache_model_name=cache_model_name,
            )
            preds_oop, preds_proba_oop = model_pl.predict(
                df_oop,
                predict_from_cached_fit=False,
                fit=best_fit,
                cache_model_name=cache_model_name,
            )

            # Evaluate results
            acc_oos = accuracy_score(df_oos["churn"], preds_oos)
            prec_oos = precision_score(df_oos["churn"], preds_oos)
            rec_oos = recall_score(df_oos["churn"], preds_oos)
            f1_oos = f1_score(df_oos["churn"], preds_oos)
            auroc_oos = roc_auc_score(df_oos["churn"], preds_proba_oos)
            auprc_oos = average_precision_score(df_oos["churn"], preds_proba_oos)

            acc_oop = accuracy_score(df_oop["churn"], preds_oop)
            prec_oop = precision_score(df_oop["churn"], preds_oop)
            rec_oop = recall_score(df_oop["churn"], preds_oop)
            f1_oop = f1_score(df_oop["churn"], preds_oop)
            auroc_oop = roc_auc_score(df_oop["churn"], preds_proba_oop)
            auprc_oop = average_precision_score(df_oop["churn"], preds_proba_oop)

            # Cache results
            cache_dict[cache_model_name] = {
                'sampling': f"{sampling_method}: {sampling_dict['sampling'][sampling_method]}",
                'dr_method': f"{dr_method}",
                'features_after_dr': list(df_train_dr.columns),
                'loss': {'alpha': cl_alpha, 'gamma': cl_gamma},
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
                'path': f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/lgbm_fits/lgbm_fit_{cache_model_name}.joblib"
            }
    cache_df = pd.DataFrame.from_dict(cache_dict, orient='index').sort_values(by=['Precision_OOP', 'Precision_OOS'], ascending=False)
    cache_df.to_csv(f'lgbm_results/results_a{cl_alpha}_g{cl_gamma}.csv')
    return cache_df



sampling_dict = {
    'sampling': {
        'none': True,
        'down1': 0.1,
        'down2': 0.5,
        'down3': "balanced",
        'up1': 10,
        'up2': 50,
        'up3': "balanced",
        'smote': True,
    },
    'dr_method': ['no_quot', 'best_quot'],
}

alpha = [0.6, 0.7, 0.8]
gamma = [None]
seed = 16
for g in gamma:
    for a in alpha:
        if a is None and g is None:
            continue
        print(f"Alpha:{a}, Gamma:{g}")
        cache_df = create_fits_and_predictions(
            sampling_dict=sampling_dict,
            cl_alpha=a,
            cl_gamma=g,
            seed=seed,
            feature_set_from_last_fits=True,
        )