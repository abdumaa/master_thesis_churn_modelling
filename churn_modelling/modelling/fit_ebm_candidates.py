import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import os
import glob
from joblib import load
from ebm import EBM

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform



def create_fits_and_eval(sampling_dict, seed=123, feature_set_from_last_fits=False):
    """Create Fits for different sampling methods with EBMClassifier."""

    # Load raw flatfiles
    df_train = pd.read_csv('../data/toydata_trainval.csv', index_col=0)
    df_oos = pd.read_csv('../data/toydata_oos.csv', index_col=0)
    df_oop = pd.read_csv('../data/toydata_oop.csv', index_col=0)

    # Call model and split df into train and val
    model_pl = EBM(df=df_train, target="churn", test_size=0.1)

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
                "/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/*"  # find solution for that # noqa
            )
            list_of_fits_filtered = [x for x in list_of_fits if sampling_method in x.rsplit("ebm_fit_", 1)[1]]
            latest_file = max(list_of_fits_filtered, key=os.path.getctime)
            ebm_laod = load(latest_file)
            best_feats = ebm_laod.feature_name_
            best_feats.append("churn")
        else:
            best_feats = model_pl.get_best_quot_features(
                df_to_dimreduce=df_train_sampled
            )
        quot_feats, fix_features = model_pl.split_quotation_fix_features()

        for dr_method in sampling_dict['dr_method']:
            if dr_method == 'no_quot':
                fix_features.append("churn")
                features_used = fix_features
            if dr_method == 'best_quot':
                features_used = best_feats
            df_train_dr = df_train_sampled[features_used]
            features_used.remove("churn")

            ### Fit models ###
            # EBM hyperparameter dictionaries
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
                "min_samples_leaf": sp_randint(2, 5),
                "max_leaves": sp_randint(2, 5),
            }
            rscv_params = {
                "n_iter": 3,
                "random_state": seed,
                "n_jobs": -1,
                "cv": 3,
                "verbose": 100,
            }
            cache_model_name = f"ebm_{sampling_method}_{dr_method}"
            best_fit = model_pl.fit_ebm(
                df_train=df_train_dr,
                hp_fix_dict=hp_fix_dict,
                hp_tune_dict=hp_tune_dict,
                rscv_params=rscv_params,
                seed=seed,
                reduce_df_mem=True,
                save_model=True,
                cache_model_name=cache_model_name,
            )

            # Predict oos and oop
            preds_oos, preds_proba_oos = model_pl.predict(
                df_oos[features_used],
                predict_from_cached_fit=False,
                fit=best_fit,
                cache_model_name=cache_model_name,
            )
            preds_oop, preds_proba_oop = model_pl.predict(
                df_oop[features_used],
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
                'path': f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/modelling/ebm_fits/ebm_fit_{cache_model_name}.joblib"
            }
    cache_df = pd.DataFrame.from_dict(cache_dict, orient='index').sort_values(by=['AUPRC_OOP', 'AUPRC_OOS'], ascending=False)
    cache_df.to_csv(f'ebm_results/results.csv')
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

seed=10
cache_df = create_fits_and_eval(
    sampling_dict=sampling_dict,
    seed=seed,
    feature_set_from_last_fits=False,
)
