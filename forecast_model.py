#!/usr/bin/env python3
# forecast_model.py
# Train a recursive time-series forecaster (skforecast) on the monthly Chicago crimes panel.

import os

import joblib
import pandas as pd
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

KEEP = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]

# Load the prepared monthly panel produced by prep_data.py
df_monthly = pd.read_csv("data/crimes_monthly.csv", parse_dates=["date"], low_memory=False)

# Split into the three working sets by TTV flag, keeping only the modelling columns
df_train    = df_monthly.loc[df_monthly["TTV"] == "train",    KEEP].reset_index(drop=True)
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", KEEP].reset_index(drop=True)
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", KEEP].reset_index(drop=True)

print(f"df_train    : {len(df_train):,} records")
print(f"df_validate : {len(df_validate):,} records")
print(f"df_forecast : {len(df_forecast):,} records")

print("\n5 random records from df_train:")
with pd.option_context("display.max_columns", None, "display.width", None):
    print(df_train.sample(5, random_state=7).to_string(index=False))


# ---------- Train multi-series recursive forecaster ----------
# Reshape df_train into the (date x series_id) wide panel skforecast expects
df_hist = df_train.sort_values(["ward", "primary_type", "date"]).copy()
df_hist["series_id"] = df_hist["ward"].astype(str) + "_" + df_hist["primary_type"]

# series: each column is one (ward, primary_type) time series of count_0
series = df_hist.pivot(index="date", columns="series_id", values="count_0")
series.index.freq = "MS"  # monthly-start; tells skforecast the panel is regular

# Exogenous features shared across every series: just calendar year & month
exog = (
    df_hist.groupby("date")
    .agg(year=("year", "first"), month=("month", "first"))
)
exog.index.freq = "MS"

print(f"\nseries panel: {series.shape[0]} dates x {series.shape[1]} series")
print(f"exog shape:   {exog.shape}   columns: {list(exog.columns)}")
print(f"history span: {series.index.min().date()} -> {series.index.max().date()}")

# XGBoost regressor underneath the recursive forecaster
regressor = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# Recursive multi-series forecaster with 6 autoregressive lags
forecaster = ForecasterRecursiveMultiSeries(regressor=regressor, lags=6)

print("\nFitting forecaster...")
forecaster.fit(series=series, exog=exog)
print("Fit complete.")

# Persist
os.makedirs("models", exist_ok=True)
model_path = "models/forecaster.joblib"
joblib.dump(forecaster, model_path)
print(f"Saved forecaster to {model_path}")
