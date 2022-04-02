def to_categorical(df):
    """Encode all object columns to categorical for lgbm."""
    cat_features = []
    for col in df.columns:
        if df[col].dtype.name == "object" or df[col].dtype.name == "category":
            cat_features.append(col)
            df[col] = df[col].astype("category")

    # how to handle missing values
    return df, cat_features