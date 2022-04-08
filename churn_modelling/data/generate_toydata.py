import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


def generate_toydata(p_churn=0.01, len=100000, to_csv=True):
    """Generate Toydata which approx represents original data."""
    sim_df = pd.DataFrame()
    sim_df["storno"] = np.random.binomial(
        1, p_churn, len
    )
    storno_0_n_requests_1 = np.random.poisson(
        0.05, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_1 = np.random.poisson(
        0.3, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_1"] = np.where(
        sim_df["storno"] == 1, storno_1_n_requests_1, storno_0_n_requests_1
    )
    storno_0_n_requests_2 = np.random.poisson(
        0.05, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_2 = np.random.poisson(
        0.2, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_2"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_1"] + storno_1_n_requests_2,
        sim_df["n_requests_1"] + storno_0_n_requests_2,
    )
    storno_0_n_requests_3 = np.random.poisson(
        0.05, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_3 = np.random.poisson(
        0.1, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_3"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_2"] + storno_1_n_requests_3,
        sim_df["n_requests_2"] + storno_0_n_requests_3,
    )
    storno_0_n_requests_6 = np.random.poisson(
        0.15, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_6 = np.random.poisson(
        0.15, len
    ) * np.random.negative_binomial(1, 0.05, len)
    sim_df["n_requests_6"] = np.where(
        sim_df["storno"] == 1,
        sim_df["n_requests_3"] + storno_1_n_requests_6,
        sim_df["n_requests_3"] + storno_0_n_requests_6,
    )
    storno_0_n_requests_12 = np.random.poisson(
        0.3, len
    ) * np.random.negative_binomial(1, 0.05, len)
    storno_1_n_requests_12 = np.random.poisson(
        0.3, len
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
    sim_df["diff_avg_vjnbe_requests_1"] = np.where(
        sim_df["n_requests_1"] != 0,
        np.where(
            sim_df["storno"] == 1,
            30 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    sim_df["diff_avg_vjnbe_requests_2"] = np.where(
        sim_df["n_requests_2"] != 0,
        np.where(
            sim_df["storno"] == 1,
            25 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    sim_df["diff_avg_vjnbe_requests_3"] = np.where(
        sim_df["n_requests_3"] != 0,
        np.where(
            sim_df["storno"] == 1,
            20 + 15 * np.random.randn(len),
            10 + 10 * np.random.randn(len),
        ),
        0,
    )
    storno_0_other_hsntsn_requests_1 = np.random.normal(
        0.2, 0.3, len
    )
    storno_1_other_hsntsn_requests_1 = np.random.normal(
        0.3, 0.5, len
    )
    storno_0_other_hsntsn_requests_1 = np.where(
        storno_0_other_hsntsn_requests_1 >= 0,
        np.where(
            storno_0_other_hsntsn_requests_1 <= 1,
            storno_0_other_hsntsn_requests_1,
            1
        ),
        0
    )
    storno_1_other_hsntsn_requests_1 = np.where(
        storno_1_other_hsntsn_requests_1 >= 0,
        np.where(
            storno_1_other_hsntsn_requests_1 <= 1,
            storno_1_other_hsntsn_requests_1,
            1
        ),
        0
    )
    sim_df["other_hsntsn_requests_1"] = np.where(
        sim_df["n_requests_1"] != 0,
        np.where(
            sim_df["storno"] == 1,
            storno_1_other_hsntsn_requests_1,
            storno_0_other_hsntsn_requests_1,
        ),
        0,
    )
    storno_0_other_hsntsn_requests_2 = np.random.normal(
        0.2, 0.3, len
    )
    storno_1_other_hsntsn_requests_2 = np.random.normal(
        0.3, 0.5, len
    )
    storno_0_other_hsntsn_requests_2 = np.where(
        storno_0_other_hsntsn_requests_2 >= 0,
        np.where(
            storno_0_other_hsntsn_requests_2 <= 1,
            storno_0_other_hsntsn_requests_2,
            1
        ),
        0
    )
    storno_1_other_hsntsn_requests_2 = np.where(
        storno_1_other_hsntsn_requests_2 >= 0,
        np.where(
            storno_1_other_hsntsn_requests_2 <= 1,
            storno_1_other_hsntsn_requests_2,
            1
        ),
        0
    )
    sim_df["other_hsntsn_requests_2"] = np.where(
        sim_df["n_requests_2"] != 0,
        np.where(
            sim_df["storno"] == 1,
            storno_1_other_hsntsn_requests_2,
            storno_0_other_hsntsn_requests_2,
        ),
        0,
    )
    storno_0_other_hsntsn_requests_3 = np.random.normal(
        0.2, 0.3, len
    )
    storno_1_other_hsntsn_requests_3 = np.random.normal(
        0.3, 0.5, len
    )
    storno_0_other_hsntsn_requests_3 = np.where(
        storno_0_other_hsntsn_requests_3 >= 0,
        np.where(
            storno_0_other_hsntsn_requests_3 <= 1,
            storno_0_other_hsntsn_requests_3,
            1
        ),
        0
    )
    storno_1_other_hsntsn_requests_3 = np.where(
        storno_1_other_hsntsn_requests_3 >= 0,
        np.where(
            storno_1_other_hsntsn_requests_3 <= 1,
            storno_1_other_hsntsn_requests_3,
            1
        ),
        0
    )
    sim_df["other_hsntsn_requests_3"] = np.where(
        sim_df["n_requests_3"] != 0,
        np.where(
            sim_df["storno"] == 1,
            storno_1_other_hsntsn_requests_3,
            storno_0_other_hsntsn_requests_3,
        ),
        0,
    )
    storno_0_n_accident = np.random.poisson(
        0.2, len
    )
    storno_1_n_accident = np.random.poisson(
        0.3, len
    )
    sim_df["n_accident"] = np.where(
        sim_df["storno"] == 1, storno_1_n_accident, storno_0_n_accident
    )
    storno_0_sum_accident_cost = np.random.poisson(
        10, len
    ) * np.random.negative_binomial(10, 0.05, len)
    storno_1_sum_accident_cost = np.random.poisson(
        10, len
    ) * np.random.negative_binomial(11, 0.05, len)
    sim_df["sum_accident_cost"] = np.where(
        sim_df["n_accident"] != 0,
        np.where(
            sim_df["storno"] == 1,
            storno_1_sum_accident_cost,
            storno_0_sum_accident_cost,
        ),
        0,
    )
    storno_0_vehicle_age = np.random.poisson(
        2.5, len
    ) * np.random.negative_binomial(15, 0.8, len)
    storno_1_vehicle_age = np.random.poisson(
        3, len
    ) * np.random.negative_binomial(15, 0.8, len)
    sim_df["vehicle_age"] = np.where(
        sim_df["storno"] == 1, storno_1_vehicle_age, storno_0_vehicle_age
    )
    storno_0_diff_car_holder = np.random.binomial(
        1, 0.2, len
    )
    storno_1_diff_car_holder = np.random.binomial(
        1, 0.4, len
    )
    sim_df["diff_car_holder"] = np.where(
        sim_df["storno"] == 1, storno_1_diff_car_holder, storno_0_diff_car_holder
    )
    sim_df["contract_age_months"] = np.random.poisson(
        8, len
    ) * np.random.negative_binomial(15, 0.7, len)
    storno_0_age_contract_holder = np.random.negative_binomial(
        75, 0.68, len
    )
    storno_1_age_contract_holder = np.random.negative_binomial(
        65, 0.68, len
    )
    sim_df["age_contract_holder"] = np.where(
        sim_df["storno"] == 1,
        np.where(
            storno_1_age_contract_holder >= 18,
            storno_1_age_contract_holder,
            18
        ),
        np.where(
            storno_0_age_contract_holder >= 18,
            storno_0_age_contract_holder,
            18
        ),
    )
    storno_0_age_youngest_driver = np.random.negative_binomial(
        55, 0.68, len
    )
    storno_1_age_youngest_driver = np.random.negative_binomial(
        40, 0.68, len
    )
    sim_df["age_youngest_driver"] = np.where(
        sim_df["storno"] == 1,
        np.where(
            storno_1_age_youngest_driver >= 18,
            storno_1_age_youngest_driver,
            18
        ),
        np.where(
            storno_0_age_youngest_driver >= 18,
            storno_0_age_youngest_driver,
            18
        ),
    )
    sim_df["age_youngest_driver"] = np.where(
        sim_df["age_youngest_driver"] > sim_df["age_contract_holder"],
        sim_df["age_contract_holder"],
        sim_df["age_youngest_driver"]
    )
    base_age = np.random.binomial(
        1, 0.4, len
    ) + np.random.binomial(
        1, 0.4, len
    ) + np.random.binomial(
        1, 0.4, len
    ) + np.random.binomial(
        1, 0.4, len
    ) + 18
    sim_df["years_driving_license"] = np.where(
        sim_df["age_contract_holder"] - base_age >= 0,
        sim_df["age_contract_holder"] - base_age,
        0
    )

    # Drop helper columns
    sim_df = sim_df.drop(
        [
            "avg_n_requests_1",
            "avg_n_requests_2",
            "avg_n_requests_3",
            "n_requests_6",
            "n_requests_12",
        ],
        axis=1
    )

    if to_csv:
        return sim_df.to_csv('toydata.csv')
    else:
        return sim_df

generate_toydata(to_csv=True)
# sns.histplot(x=df["n_requests_1"])
# df["n_requests_1"].value_counts()
# df.groupby("storno").mean()

# Playground
# x_ds = sim_df.copy()
# 
# np.mean(
#     np.random.poisson(
#         8, len(sim_df)
#     ) * np.random.negative_binomial(15, 0.7, len(sim_df))
# )
sns.histplot(
    x=np.random.negative_binomial(
        18, 0.68, len(df)
    ) #* np.random.poisson(3, len(df))
)

# sns.histplot(x=sim_df["diff_vjnbe_avg_1"])
# sim_df.groupby("storno").mean()
# # Plots
# sns.catplot(
#     x="n_requests_1",
#     y="storno",
#     kind="box",
#     orient="h",
#     height=2.5,
#     aspect=4,
#     data=sim_df
# )