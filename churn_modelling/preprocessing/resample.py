import pandas as pd
from imblearn.over_sampling import SMOTE


def resample(df_to_sample, target="churn", sampling="down", frac="balanced", seed=1234):
    """Resample data to handle class imbalance.

    sampling: str
        "down" or "up" or "smote"
    frac: str/float
        "balanced" or frac
    """
    if sampling == "down":
        resample_class = 0
    elif sampling == "up" or sampling == "smote":
        resample_class = 1
    else:
        raise ValueError(f"Sampling method {sampling} not supported")

    to_be_resampled = len(df_to_sample.query(f"{target} == {resample_class}"))
    base = len(df_to_sample.query(f"{target} != {resample_class}"))
    if frac == "balanced":
        frac = base / to_be_resampled

    if sampling == "smote":
        ratio = (to_be_resampled*frac)/base
        sm = SMOTE(
            sampling_strategy=ratio,
            random_state=seed,
            k_neighbors=5,
            n_jobs=-1,
            )
        y = df_to_sample[f"{target}"]
        X = df_to_sample.drop([f"{target}"], axis=1)
        df_res, y_res = sm.fit_resample(X, y)
        df_res[f"{target}"] = y_res
        return df_res
    else:
        return pd.concat(
            [
                df_to_sample.query(f"{target} == {resample_class}").sample(
                    frac=frac, replace=True, random_state=seed
                ),
                df_to_sample.query(f"{target} != {resample_class}"),
            ],
            sort=False,
            ignore_index=True,
        )