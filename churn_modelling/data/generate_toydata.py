import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


def generate_toydata(p_churn=0.1, len=100000, to_csv=True):
    """Generate Toydata which approx represents original data."""
    sim_df = pd.DataFrame()
    sim_df["storno"] = np.random.binomial(
        1, p_churn, len
    )
    storno_0_n_requests_1 = np.random.poisson(
        0.011, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_1 = np.random.poisson(
        0.11, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_1"] = np.where(
        sim_df["storno"] == 1, storno_1_n_requests_1, storno_0_n_requests_1
    )
    storno_0_n_requests_2 = np.random.poisson(
        0.011, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_2 = np.random.poisson(
        0.04, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_2"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_1"] + storno_1_n_requests_2,
        sim_df["n_requests_1"] + storno_0_n_requests_2,
    )
    storno_0_n_requests_3 = np.random.poisson(
        0.011, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_3 = np.random.poisson(
        0.02, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_3"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_2"] + storno_1_n_requests_3,
        sim_df["n_requests_2"] + storno_0_n_requests_3,
    )
    storno_0_n_requests_6 = np.random.poisson(
        0.033, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_6 = np.random.poisson(
        0.033, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_6"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_3"] + storno_1_n_requests_6,
        sim_df["n_requests_3"] + storno_0_n_requests_6,
    )
    storno_0_n_requests_12 = np.random.poisson(
        0.066, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_12 = np.random.poisson(
        0.066, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_12"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_6"] + storno_1_n_requests_12,
        sim_df["n_requests_6"] + storno_0_n_requests_12,
    )
    sim_df["avg_n_requests_1"] = round(
        (sim_df["n_requests_12"] - sim_df["n_requests_1"]) / 11, 2
    )
    sim_df["avg_n_requests_2"] = round(
        (sim_df["n_requests_12"] - sim_df["n_requests_2"]) / 10, 2
    )
    sim_df["avg_n_requests_3"] = round(
        (sim_df["n_requests_12"] - sim_df["n_requests_3"]) / 9, 2
    )
    sim_df["diff_n_requests_1"] = sim_df["n_requests_1"] - sim_df["avg_n_requests_1"]
    sim_df["diff_n_requests_2"] = (
        round(sim_df["n_requests_2"] / 2, 2) - sim_df["avg_n_requests_2"]
    )
    sim_df["diff_n_requests_3"] = (
        round(sim_df["n_requests_3"] / 3, 2) - sim_df["avg_n_requests_3"]
    )
    sim_df["diff_vjnbe_avg_1"] = np.where(
        sim_df["n_requests_1"] != 0,
        np.where(
            sim_df["storno"] == 1,
            30 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    sim_df["diff_vjnbe_avg_2"] = np.where(
        sim_df["n_requests_2"] != 0,
        np.where(
            sim_df["storno"] == 1,
            30 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    sim_df["diff_vjnbe_avg_3"] = np.where(
        sim_df["n_requests_3"] != 0,
        np.where(
            sim_df["storno"] == 1,
            30 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    storno_0_n_unfall = np.random.poisson(
        0.2, len
    )
    storno_1_n_unfall = np.random.poisson(
        0.3, len
    )
    sim_df["n_unfall"] = np.where(
        sim_df["storno"] == 1, storno_1_n_unfall, storno_0_n_unfall
    )
    storno_0_sum_aufwand = np.random.poisson(
        10, len
    ) * np.random.negative_binomial(10, 0.05, len)
    storno_1_sum_aufwand = np.random.poisson(
        13, len
    ) * np.random.negative_binomial(12, 0.05, len)
    sim_df["sum_aufwand"] = np.where(
        sim_df["n_unfall"] != 0,
        np.where(
            sim_df["storno"] == 1,
            storno_1_sum_aufwand,
            storno_0_sum_aufwand,
        ),
        0,
    )
    storno_0_fzg_alter = np.random.poisson(
        2.5, len
    ) * np.random.negative_binomial(15, 0.8, len)
    storno_1_fzg_alter = np.random.poisson(
        3, len
    ) * np.random.negative_binomial(15, 0.8, len)
    sim_df["fzg_alter"] = np.where(
        sim_df["storno"] == 1, storno_1_fzg_alter, storno_0_fzg_alter
    )
    storno_0_abw_halter = np.random.binomial(
        1, 0.2, len
    )
    storno_1_abw_halter = np.random.binomial(
        1, 0.4, len
    )
    sim_df["abw_halter"] = np.where(
        sim_df["storno"] == 1, storno_1_abw_halter, storno_0_abw_halter
    )
    sim_df["vbeg_alter_months"] = np.random.poisson(
        8, len
    ) * np.random.negative_binomial(15, 0.7, len)

    if to_csv:
        return sim_df.to_csv('toydata.csv')
    else:
        return sim_df

generate_toydata()

sim_df.columns

# Playground
y_ds = X_ds["diff_vjnbe_avg_1"]
x_ds = sim_df.copy()

lr = LinearRegression(n_jobs=-1).fit(x_ds, y_ds)
lr.coef_
lr.intercept_
x_ds["n_requests_1_fit_raw"] = round(lr.predict(x_ds), 2)

np.mean(
    np.random.poisson(
        8, len(sim_df)
    ) * np.random.negative_binomial(15, 0.7, len(sim_df))
)
sns.histplot(
    x=np.random.poisson(
        8, len(sim_df)
    ) * np.random.negative_binomial(15, 0.7, len(sim_df))
)
sns.histplot(x=X_ds.loc[X_ds["storno"] == 0]["as_months_vbeg_alter"])
np.mean(X_ds.loc[X_ds["storno"] == 0]["as_months_vbeg_alter"])
sns.histplot(x=sim_df["diff_vjnbe_avg_1"])
sim_df.groupby("storno").mean()
# Plots
sns.catplot(
    x="n_requests_1",
    y="storno",
    kind="box",
    orient="h",
    height=2.5,
    aspect=4,
    data=X_ds,
)