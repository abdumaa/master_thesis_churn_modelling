import pandas as pd


def resample(df_to_sample, y="storno", sampling="down", frac="balanced", seed=1234):
    """Resample data to handle class imbalance."""
    if sampling == "down":
        resample_class = 0
    elif sampling == "up":
        resample_class = 1
    # elif sampling == "smote":
    #     resample_class = 1
    else:
        raise ValueError(f"Sampling method {sampling} not supported")

    if frac == "balanced":
        to_be_resampled = len(df_to_sample.query(f"{y} == {resample_class}"))
        base = len(df_to_sample.query(f"{y} != {resample_class}"))
        frac = base / to_be_resampled

    return pd.concat(
        [
            df_to_sample.query(f"{y} == {resample_class}").sample(
                frac=frac, replace=False, random_state=seed
            ),
            df_to_sample.query(f"{y} != {resample_class}"),
        ],
        sort=False,
        ignore_index=True,
    )