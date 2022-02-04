import pandas as pd
from sklearn.model_selection import train_test_split


empty_df = pd.DataFrame()


def split_train_test(df=empty_df, X=empty_df, y=empty_df, test_size=0.01, seed=1234):
    """Split df in Train and Test.

    Either give df or X and y as inputs.
    """
    if not X.empty and not y.empty and df.empty:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    elif not df.empty and X.empty and y.empty:
        df_test = df.sample(frac=test_size, random_state=seed)
        df_train = df.drop(df_test.index)
        return df_train, df_test
    else:
        raise ValueError("Either define only df or only X and y")