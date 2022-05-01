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
│   │   ├── utils.py
│   │   ├── ebm_fits
│   │   ├── ebm_results
│   │   ├── init_scores
│   │   ├── lgbm_fits
│   │   └── lgbm_results
│   ├── playground
│   │   ├── ebm_explainability.ipynb
│   │   ├── ebm_fit_candidates.ipynb
│   │   ├── gbt_explainability.ipynb
│   │   ├── gbt_fit_candidates.ipynb
│   │   ├── simulations.ipynb
│   │   ├── ebm_fits
│   │   ├── ebm_results
│   │   ├── init_scores
│   │   ├── lgbm_fits
│   │   ├── lgbm_results
│   │   └── plots
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