# Code Repository for Master Thesis: "Predicting Customer Churn using Quotation Data"

This repository contains the entire coding, data and tex files used for this master thesis. For inspection purposes please make sure to follow the guidelines listet below.

# Repository Structure
```bash
.
├── README.md
├── churn_modelling
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataviz.py
│   │   ├── toydata.csv
│   │   ├── toydata_oop.csv
│   │   ├── toydata_oos.csv
│   │   └── toydata_trainval.csv
│   ├── modelling
│   │   ├── __init__.py
│   │   ├── custom_loss.py
│   │   ├── ebm.py
│   │   ├── lgbm.py
│   │   ├── predict_simulate.py
│   │   ├── ebm_fits
│   │   │   └── ebm_fit_ebm_down2_best_quot.joblib
│   │   ├── ebm_results
│   │   │   └── results.csv
│   │   ├── init_scores
│   │   ├── lgbm_fits
│   │   │   └── lgbm_fit_gbt_up1_best_quot_aNone_gNone.joblib
│   │   ├── lgbm_results
│   │   │   ├── results_a0.6_g0.2.csv
│   │   │   ├── ...
│   │   │   └── results_aNone_gNone.csv
│   │   └── utils.py
│   ├── playground
│   │   ├── ebm_explainability.ipynb
│   │   ├── ebm_fit_candidates.ipynb
│   │   ├── ebm_fits
│   │   │   ├── ebm_fit_ebm_down1_best_quot.joblib
│   │   │   ├── ...
│   │   │   └── test.joblib
│   │   ├── ebm_results
│   │   │   └── results.csv
│   │   ├── gbt_explainability.ipynb
│   │   ├── gbt_fit_candidates.ipynb
│   │   ├── init_scores
│   │   │   ├── lgbm_fit_gbt_down1_best_quot_a0.6_g0.2.joblib
│   │   │   ├── ...
│   │   │   └── lgbm_fit_gbt_down2_no_quot_a0.7_g0.5.joblib
│   │   ├── lgbm_fits
│   │   │   ├── lgbm_fit_gbt_down1_best_quot_a0.6_gNone.joblib
│   │   │   ├── ...
│   │   │   └── test.joblib
│   │   ├── lgbm_results
│   │   │   └── results.csv
│   │   ├── plots
│   │   │   ├── ebm_prec_rec_plot.png
│   │   │   ├── ...
│   │   │   └── shape_function_n_requests_1.png
│   │   └── simulations.ipynb
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── categorical.py
│   │   ├── mrmr.py
│   │   ├── resample.py
│   │   └── splitting.py
│   └── utils
│       ├── __init__.py
│       └── mem_usage.py
├── requirements.txt
└── tex
    ├── images
    ├── literaturechurn.bib
    ├── makefile.sh
    ├── thesis.pdf
    └── thesis.tex
```