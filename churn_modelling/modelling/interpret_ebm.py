import pandas as pd
import numpy as np
import ast
import glob
from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns


def plot_shape_function(model_folder, model, feats, save=True, save_folder_path=''):
    """Plot shape function(s) k() for a given EBM model and feature(s)."""

    # Load model
    path = f"/Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/churn_modelling/{model_folder}/ebm_fits/{model}.joblib"
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
        sns.lineplot(x,y_vals, drawstyle='steps-post')
        plt.xlabel(f)
        plt.ylabel(f'$k_{idx+1}$({f})')
        if save:
            plt.savefig(f"{save_folder_path}shape_function_{f}.png", dpi=100, bbox_inches='tight')
        plt.clf()


features = [
    'n_requests_1',
    'diff_avg_vjnbe_requests_3',
    'diff_n_requests_3',
    'other_hsntsn_requests_3',
    'n_accident',
    'sum_accident_cost',
    'vehicle_age',
    'contract_age_months',
    'age_contract_holder',
    'age_youngest_driver',
    'years_driving_license',
]

plot_shape_function(
    model_folder='modelling',
    model='ebm_fit_ebm_down2_best_quot',
    feats=features,
    save_folder_path='../../tex/images/',
)