# Code Repository for Master Thesis: "Predicting Customer Churn using Quotation Data"

This repository contains the entire coding, data and tex files used for this master thesis. For inspection purposes please make sure to follow the guidelines listet below.

# Cloning & Working Mode

In order to clone the repository and to install the [dependencies](https://github.com/abdumaa/master_thesis_churn_modelling/network/dependencies) do following steps:

1. From a terminal window change to the local directory where you want to clone the repository
2. Clone the repository using git
```
      git clone https://github.com/abdumaa/master_thesis_churn_modelling.git
```
3. Enter Repository
```
      cd master_thesis_churn_modelling
```
4. Install all [dependencies](https://github.com/abdumaa/master_thesis_churn_modelling/network/dependencies) necessary for this project using pip by entering this exact command
```
      pip install -r requirements.txt
```

# Inspection

## Function Testing
In order to test the functions, I refer you to the [playground](https://github.com/abdumaa/master_thesis_churn_modelling/tree/main/churn_modelling/playground) folder. In that folder I prepared jupyter notebooks, where each and every function used in this work can be tested. You are also invited to play around with some (hyper)parameters.
I highly recommend using Jupyter to open these notebooks due to the simple User-Interface. Follow these steps to open these files on a Jupyter server.
1. Download jupyter in the terminal if it is not yet downloaded
```
      pip install notebook
```
2. Start server with this command in your terminal
```
      jupyter notebook churn_modelling/playground/
```
3. If you want to close the server repeat "Ctrl+C" twice in the terminal

## Code Review
To get a better view on the code leading to the results, please visit the folder [modelling](https://github.com/abdumaa/master_thesis_churn_modelling/tree/main/churn_modelling/modelling).


## Thesis
You can also review my thesis here which is located in the [tex](https://github.com/abdumaa/master_thesis_churn_modelling/blob/main/tex/thesis.pdf) folder.

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