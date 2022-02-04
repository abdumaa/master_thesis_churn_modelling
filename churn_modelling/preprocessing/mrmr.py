import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_validate
import scipy.stats as ss
from itertools import combinations_with_replacement, product
from .categorical import to_categorical


def _lgbm_scorer(
    df, target, cv=5,
):
    """Create relevance score for MRMR."""
    y = df[target]
    X = df.drop(target, axis=1)
    X, cat_features = to_categorical(X)

    gbt = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        min_child_weight=250,
        num_leaves=5,
        class_weight="balanced",
        n_jobs=-1,
        n_estimators=200,
        random_state=1,
        verbose=-100,
        reg_lambda=5,
        subsample=0.6,
        subsample_freq=1,
        importance_type="split",
        # categorical_feature=cat_features,
    )
    output_cv = cross_validate(
        estimator=gbt, X=X, y=y, cv=cv, n_jobs=-1, return_estimator=True, verbose=False,
    )

    importances_df = pd.DataFrame(index=X.columns)
    for idx, estimator in enumerate(output_cv["estimator"]):
        importances_df[f"estimator_{idx}"] = estimator.feature_importances_

    importances_df["avg_importance"] = importances_df.mean(axis=1)

    return importances_df.sort_values(by=["avg_importance"], ascending=False)


# Oder target encoding für cat/ordinal features?


def _cramersv_corr(x, y):
    """Get corrected Cramers-V for correlation between cat features.

    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical
    Society 42 (2013): 323-328.
    Returns a float.
    """
    # Clear Nulls
    df = pd.DataFrame(data={x.name: x, y.name: y}).dropna().reset_index(drop=True)

    # Compute confusion matrix
    if pd.Series.equals(x, y):
        conf_mat = pd.crosstab(df.iloc[:, 0], df.iloc[:, 0])
    else:
        conf_mat = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])

    # Compute corrected Cramers V
    r, k = conf_mat.shape
    if min(r, k) < 2:
        print(
            f"At least 1 Feature of {x.name}, {y.name} has Cardinality smaller 2 in Training Set, Correlation set to very small number"  # noqa
        )
        return 0.0
    else:
        chi2 = ss.chi2_contingency(conf_mat)[0]
        n = conf_mat.sum().sum()
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _pointbis_corr(x, y):
    """Get correlation between dummy x and continuous y variables.

    Returns a float.
    """
    # Clear Nulls
    df = pd.DataFrame(data={x.name: x, y.name: y}).dropna().reset_index()
    x, y = df[x.name], df[y.name]

    # Check Cardinality
    x_kar, y_kar = x.nunique(), y.nunique()
    if min(x_kar, y_kar) < 2:
        print(
            f"At least 1 Feature of {x.name}, {y.name} has Cardinality smaller 2 in Training Set, Correlation set to very small number"  # noqa
        )
        return 0.0
    else:
        return ss.pointbiserialr(x, y)[0]


def _eta_corr(x, y):
    """Get correlation between nominal x and continuous y variables.

    Returns a score.
    """
    # Clear Nulls
    df = pd.DataFrame(data={x.name: x, y.name: y}).dropna().reset_index()
    x, y = df[x.name], df[y.name]

    # Calculate eta
    fcat, _ = pd.factorize(x)
    cat_num = np.max(fcat) + 1
    x_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = y[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        x_avg_array[i] = np.average(cat_measures)
    x_total_avg = np.sum(np.multiply(x_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(x_avg_array, x_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(y, x_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    return eta


def _bravaisp_corr(df):
    """Get correlation between continuous variables.

    Returns a df.
    """
    return df.corr().fillna(0.0)


def _correlation_scorer(df):
    """Create redundancy score for MRMR."""
    # Collect different dtype columns in seperate dfs
    bool_filter = ((df == 1) | (df == 0) | (df.isna())).all()
    df_num = df.select_dtypes(exclude=["object_"]).loc[:, ~bool_filter]
    df_bool = df.loc[:, bool_filter]
    df_obj = df.select_dtypes(include=["object_"])

    # Create empty mask
    df_corr = pd.DataFrame(np.nan, index=df.columns, columns=df.columns)

    # Create correlation matrix with BP for continuous features
    df_corr = df_corr.combine_first(_bravaisp_corr(df_num))

    # Create correlation with Cramers V for cat features
    df_obj_bool = pd.concat([df_obj, df_bool], axis=1)
    combis = list(combinations_with_replacement(df_obj_bool.columns, 2))
    for combo in combis:
        cramers_v = _cramersv_corr(df_obj_bool[combo[0]], df_obj_bool[combo[1]])
        df_corr.loc[combo[0], combo[1]] = cramers_v
        df_corr.loc[combo[1], combo[0]] = cramers_v

    # Create correlation with pointbis for dummy and continuous pairs
    combis = list(product(df_bool.columns, df_num.columns))
    for combo in combis:
        pointbis = _pointbis_corr(df_bool[combo[0]], df_num[combo[1]])
        df_corr.loc[combo[0], combo[1]] = pointbis
        df_corr.loc[combo[1], combo[0]] = pointbis

    # Create correlation with eta for nominal and continuous pairs
    combis = list(product(df_obj.columns, df_num.columns))
    for combo in combis:
        eta = _eta_corr(df_obj[combo[0]], df_num[combo[1]])
        df_corr.loc[combo[0], combo[1]] = eta
        df_corr.loc[combo[1], combo[0]] = eta

    return df_corr


def mrmr(df, target, cv=5):
    """Iterate over entire feature set to get all possible best subsets."""
    # Create Correlation Matrix
    print("------------------ COMPUTE CORRELATION MATRIX")
    df_corr = _correlation_scorer(df)

    # Create empty and full basket of features
    unselected_list = df.columns.to_list()
    unselected_list.remove(target)  # exclude target feature
    selected_list = []
    iterations_df = pd.DataFrame(
        {
            "ITERATION": [0],
            "UNSELECTED_SET": [f"{unselected_list}"],
            "SELECTED_SET": [f"{selected_list}"],
            "SELECTED_FEATURE": [""],
            "MRMR_SCORE": [""],
        }
    )
    print("------------------ START ITERATING THROUGH FEATURE SET")
    for i in range(1, len(df.columns)):
        print(f"------------------ ITERATION STEP {i}")
        df_temp = pd.concat([df[unselected_list], df[target]], axis=1)
        relevance_df = _lgbm_scorer(df=df_temp, target=target, cv=cv)
        if i == 1:
            selected_feature, mrmr_score = (
                relevance_df.index[0],
                relevance_df["avg_importance"][0],
            )
        else:
            redundancy_df = (
                df_corr.loc[selected_list, unselected_list].abs().mean(axis=0)
            )
            mrmr_score_df = (
                relevance_df["avg_importance"]
                .div(redundancy_df)
                .sort_values(ascending=False)
            )
            selected_feature, mrmr_score = mrmr_score_df.index[0], mrmr_score_df.iloc[0]

        print(f"{selected_feature} selected with MRMR-Score: {mrmr_score}")
        unselected_list.remove(selected_feature)
        selected_list.append(selected_feature)
        # print("Unselected:", unselected_list)
        # print("Selected:", selected_list)
        iterations_df.loc[i] = [
            i,
            f"{unselected_list}",
            f"{selected_list}",
            selected_feature,
            mrmr_score,
        ]

    print("------------------ END ITERATING THROUGH FEATURE SET")

    return iterations_df