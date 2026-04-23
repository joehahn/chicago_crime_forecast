"""
forecast_model.py

Trains an skforecast multi-series recursive forecaster on the monthly Chicago
crime panel -- one timeseries per (ward, primary_type) -- with an XGBoost
regressor underneath. The trained forecaster is saved to
models/forecaster.joblib.
"""

import os

import pandas as pd
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import save_forecaster
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Load crimes_monthly.csv and carve out train / validate / forecast slices.
# ---------------------------------------------------------------------------

df_monthly = pd.read_csv("data/crimes_monthly.csv", parse_dates=["date"])

keep_cols = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]

df_train = df_monthly.loc[df_monthly["TTV"] == "train", keep_cols].copy()
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", keep_cols].copy()
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", keep_cols].copy()

print(f"df_train    has {len(df_train):,} records")
print(f"df_validate has {len(df_validate):,} records")
print(f"df_forecast has {len(df_forecast):,} records")

print("\n5 random records from df_train:")
print(df_train.sample(n=5, random_state=None).to_string())


# ---------------------------------------------------------------------------
# Reshape df_train into skforecast's multi-series format:
#   - `series`: dict-of-Series (or wide DataFrame) keyed by series_id, index=date
#   - `exog`:   DataFrame indexed by date with the shared exogenous features.
# ---------------------------------------------------------------------------

# one series per (ward, primary_type) combination
df_train["series_id"] = (
    "w" + df_train["ward"].astype(int).astype(str).str.zfill(2)
    + "_" + df_train["primary_type"].str.replace(" ", "_")
)

series_wide = (
    df_train.pivot_table(
        index="date",
        columns="series_id",
        values="count_0",
        aggfunc="first",
    )
    .sort_index()
    .asfreq("MS")  # ensure contiguous monthly index
)

# exogenous features — shared across all series, indexed by date
exog = (
    df_train[["date", "year", "month"]]
    .drop_duplicates()
    .set_index("date")
    .sort_index()
    .asfreq("MS")
)

print(f"\nseries_wide shape: {series_wide.shape}  (dates x series)")
print(f"exog shape: {exog.shape}")


# ---------------------------------------------------------------------------
# Build and fit the forecaster.
# ---------------------------------------------------------------------------

regressor = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

forecaster = ForecasterRecursiveMultiSeries(
    regressor=regressor,
    lags=6,
    encoding="ordinal",
)

print("\nfitting forecaster ...")
forecaster.fit(series=series_wide, exog=exog)
print("fit complete")


# ---------------------------------------------------------------------------
# Persist the forecaster.
# ---------------------------------------------------------------------------

os.makedirs("models", exist_ok=True)
save_forecaster(forecaster, file_name="models/forecaster.joblib", verbose=False)
print("\nsaved forecaster to models/forecaster.joblib")
