import pandas as pd
import numpy as np
import ast
import glob
from joblib import dump, load

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
import lime


def plot_pdp(df, model_folder, model, feats, target="churn", save=True, save_folder_path=''):
    """Plot PDP for a given model and feature."""

    # Load model
    path = f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/{model_folder}/lgbm_fits/{model}.joblib"
    model_gbt = load(path)

    # Split y and X
    features = model_gbt.feature_name_
    X = df[features]
    y = df[target]

    # Create PDP-Plots
    for f in feats:
        PartialDependenceDisplay.from_estimator(model_gbt, X, [f], target=target, n_jobs=-1)
        if save:
            plt.savefig(f"{save_folder_path}pdp_{f}.png", dpi=100, bbox_inches='tight')


df_test = pd.read_csv('../data/toydata_oos.csv', index_col=0)
model = 'lgbm_fit_gbt_up1_best_quot_aNone_gNone'
plot_pdp(
    df=df_test,
    model_folder='modelling',
    model=model,
    feats=['age_youngest_driver', 'age_contract_holder', 'years_driving_license'],
    save_folder_path='../../tex/images/',
)


def plot_shap_summary(df, model_folder, model, target="churn", save=True, save_folder_path=''):
    """Plot SHAP's summary plot for a given model."""

    # Load model
    path = f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/{model_folder}/lgbm_fits/{model}.joblib"
    model_gbt = load(path)

    # Split y and X
    features = model_gbt.feature_name_
    X = df[features]
    y = df[target]

    # Calculate shap values
    explainer = shap.TreeExplainer(model_gbt)
    shap_values = explainer.shap_values(X)

    # Plot summary plot
    fig = shap.summary_plot(shap_values[1], X, color_bar_label='Variable value', show=False)

    if save:
        plt.savefig(f"{save_folder_path}shap_summary.png", dpi=200, bbox_inches='tight')

    return fig


df_test = pd.read_csv('../data/toydata_oos.csv', index_col=0)
model = 'lgbm_fit_gbt_up1_best_quot_aNone_gNone'

plot_shap_summary(
    df=df_test,
    model_folder='modelling',
    model=model,
    save_folder_path='../../tex/images/',
)

def plot_shap_waterfall(df, model_folder, model, idx, target="churn", save=True, save_folder_path=''):
    """Plot SHAP's summary plot for a given model."""

    # Load model
    path = f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/{model_folder}/lgbm_fits/{model}.joblib"
    model_gbt = load(path)

    # Split y and X
    features = model_gbt.feature_name_
    X = df[features]
    y = df[target]

    # Calculate shap values
    explainer = shap.Explainer(model_gbt, X)
    shap_values = explainer(X.loc[[idx]])

    # Plot summary plot
    shap.plots.waterfall(shap_values[0], show=True)

    if save:
        plt.gcf()
        plt.savefig(f"{save_folder_path}shap_waterfall_example.png", dpi=200, bbox_inches='tight')


df_test = pd.read_csv('../data/toydata_oos.csv', index_col=0)
model = 'lgbm_fit_gbt_up1_best_quot_aNone_gNone'

plot_shap_waterfall(
    df=df_test,
    model_folder='modelling',
    model=model,
    idx=78790,
    save_folder_path='../../tex/images/',
)
