import pandas as pd
from lgbm import LGBM


def create_predictions():  # maybe include dim_reduction as param here?
    """Create Predictions."""

    # Load raw flatfile
    df_temp = pd.read_csv('../data/toydata_active.csv', index_col=0)
    df = df_temp.copy()

    # Create predictions and store in df
    model_pl = LGBM(df=df, target="storno")
    preds, preds_proba = model_pl.predict(
        X=df, predict_from_latest_fit=True, lgbm_fit=None, reduce_df_mem=True,
    )
    df["storno_prob"] = preds_proba
    df["storno"] = preds

    return df