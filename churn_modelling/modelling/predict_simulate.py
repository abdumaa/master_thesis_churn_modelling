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
from ebm import EBM
from lgbm import LGBM


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
    if cache_model_name[:3] == 'gbt':
        model_pl = LGBM(df=df_test, target="churn")
    elif cache_model_name[:3] == 'ebm':
        model_pl = EBM(df=df_test, target="churn")
    else:
        ValueError(
            f"{cache_model_name} should either start with 'gbt' or with 'ebm'"
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
    if save:
        plt.savefig(f"{save_folder_path}{cache_model_name[:3]}_prec_rec_plot.png", dpi=100, bbox_inches='tight')

create_prec_rec_plot(
    model_folder='modelling',
    cache_model_name='gbt_up1_best_quot_aNone_gNone',
    save_folder_path='../../tex/images/',
)