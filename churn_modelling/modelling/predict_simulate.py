import pandas as pd
import numpy as np
import ast
import glob
from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from churn_modelling.modelling.ebm import EBM
from churn_modelling.modelling.lgbm import LGBM



# Best models of each modelling approach
model_candidates = [
    'gbt_up1_best_quot_aNone_gNone',
    'ebm_down2_best_quot',
]


def create_prec_rec_plot(model_folder, cache_model_name, save=True, save_folder_path=''):
    """Plot Recall and Precision of model for different cutoffs."""

    # Load Unseen Data
    df_test = pd.read_csv('../data/toydata_oos.csv', index_col=0)
    X_test = df_test.drop(["churn"], axis=1)
    y_test = df_test["churn"]

    # Load model
    if cache_model_name[:3] == 'lgb':
        model_pl = LGBM()
    elif cache_model_name[:3] == 'ebm':
        model_pl = EBM()
    else:
        ValueError(
            f"{cache_model_name} should either start with 'lgb' or with 'ebm'"
        )

    # Create predictions
    preds, preds_proba = model_pl.predict(
        X=X_test,
        predict_from_cached_fit=True,
        cache_model_name=cache_model_name,
    )

    # Simulate Precision, Recall and F1-Score for tau space in [0, 1]
    tau_space = np.linspace(0, 1, 1001)
    prec = np.empty_like(tau_space, dtype=float)
    rec = np.empty_like(tau_space, dtype=float)
    f1 = np.empty_like(tau_space, dtype=float)
    for i, tau in enumerate(tau_space):
        preds_new = (preds_proba >= tau).astype("int")
        prec[i] = precision_score(y_test, preds_new, zero_division=1)
        rec[i] = recall_score(y_test, preds_new)
        f1[i] = f1_score(y_test, preds_new)

    # Store results in df
    df_evalf = pd.DataFrame(
        {'tau': tau_space, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    )
    df_eval = df_evalf.melt(
        id_vars=['tau'],
        value_name='Score',
        var_name='Metric',
    )

    # Create plot
    sns.lineplot(x='tau', y='Score', hue='Metric', data=df_eval)
    plt.xlabel('$\mathregular{\u03C4}$')
    if save:
        plt.savefig(f"{save_folder_path}{cache_model_name[:3]}_prec_rec_plot.png", dpi=200, bbox_inches='tight')


def create_profit_simulation_plot(model_folder, cache_model_name, p_succ=0.9, NPV=100, NLV=10, K_RS=2, save=True, save_folder_path=''):
    """Plot Profit Increase of model for different cutoffs."""

    # Load Unseen Data
    df_test = pd.read_csv('../data/toydata_oos.csv', index_col=0)
    X_test = df_test.drop(["churn"], axis=1)
    y_test = df_test["churn"]

    # Load model
    if cache_model_name[:3] == 'lgb':
        model_pl = LGBM()
    elif cache_model_name[:3] == 'ebm':
        model_pl = EBM()
    else:
        ValueError(
            f"{cache_model_name} should either start with 'lgb' or with 'ebm'"
        )

    # Create predictions
    preds, preds_proba = model_pl.predict(
        X=X_test,
        predict_from_cached_fit=True,
        cache_model_name=cache_model_name,
    )

    # Simulate Precision and number of predicted churns for tau space in [0, 1]
    tau_space = np.linspace(0.1, 1, 901)
    prec = np.empty_like(tau_space, dtype=float)
    n_churn = np.empty_like(tau_space, dtype=int)
    del_profit = np.empty_like(tau_space, dtype=float)
    for i, tau in enumerate(tau_space):
        preds_new = (preds_proba >= tau).astype("int")
        prec[i] = precision_score(y_test, preds_new, zero_division=1)
        n_churn[i] = (preds_new == 1).sum()
        del_profit[i] = n_churn[i] * ((prec[i] * p_succ * (NPV - NLV)) - ((1 - prec[i]) * NLV) - K_RS)

    # Get tau which maximizes profit
    idx_opt = del_profit.argmax()
    tau_opt = tau_space[idx_opt]
    profit_opt = del_profit[idx_opt]

    # Store results in df
    df_prof = pd.DataFrame(
        {'tau': tau_space, 'Profit-Increase': del_profit}
    )

    # Create plot
    sns.lineplot(x='tau', y='Profit-Increase', data=df_prof)
    plt.ylim(plt.ylim()[0]+500, plt.ylim()[1]+500)
    plt.xlabel('$\mathregular{\u03C4}$')
    plt.vlines(tau_opt, ymin=plt.ylim()[0], ymax=profit_opt, color='black', linestyle='--', lw=0.5)
    plt.hlines(profit_opt, xmin=0, xmax=tau_opt, color='black', linestyle='--', lw=0.5)
    plt.xlim(0.1, 1)
    plt.text(x=tau_opt+0.01, y=profit_opt+80, s='$\mathregular{\u03C4^{*}=}$'+f"{tau_opt}\n"+'$\mathregular{\Delta \Pi^{*}=}$'+f"{round(profit_opt,1)}")
    if save:
        plt.savefig(f"{save_folder_path}{cache_model_name[:3]}_prof_increase_plot.png", dpi=200, bbox_inches='tight')