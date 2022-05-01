import pandas as pd
from joblib import load


def split_quotation_fix_features(df, target="churn"):
    """Return two lists containing quotation & fix features.

    Args:
        df (pandas.DataFrame): Data set containing features
        target (str, optional): Name of target variable. Defaults to "churn".

    Returns:
        tuple: Two lists (quotation, fix) of features.
    """

    full_features = df.columns.to_list()
    full_features.remove(target)
    quotation_features = [i for i in full_features if "requests" in i]
    fix_features = [i for i in full_features if "requests" not in i]

    return quotation_features, fix_features


def get_featureset_from_fit(fit, target="churn", include_target=True, include_interactions=False):
    """Extract used feature set from specific fit."""

    features_used = fit.feature_names
    single_features = []
    interaction_features = []
    for f in features_used:
        if ' x ' in f:
            interaction_features.append(f)
        else:
            single_features.append(f)
    if include_target:
        single_features.append(target)
    if include_interactions:
        return single_features, interaction_features
    else:
        return single_features


def get_featureset_from_cached_fit(
    cached_model_name,
    path_to_folder,
    target="churn",
    include_target=True,
    include_interactions=False
    ):
    """Extract used feature set from cached fit."""

    path = f"{path_to_folder}/{cached_model_name}.joblib"
    model = load(path)

    return get_featureset_from_fit(
        fit=model,
        target=target,
        include_target=include_target,
        include_interactions=include_interactions,
        )