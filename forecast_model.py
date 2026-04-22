#!/usr/bin/env python3
"""Train an skforecast recursive multi-series forecaster on monthly Chicago crime data.

Reads  : data/crimes_monthly.csv
Writes : data/crimes_train.csv
         data/crimes_validate.csv
         data/crimes_forecast.csv
         models/forecaster.joblib

Approach — one timeseries per (ward, primary_type), count_0 as the target.
Fit ONCE on all history before 2025-01-01 (TTV = 'train').
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from skforecast.recursive import ForecasterRecursiveMultiSeries

SEED = 42
LAGS = 6
CUTOFF_DATE = pd.Timestamp("2025-01-01")

ROOT = Path(__file__).parent
MONTHLY_PATH = ROOT / "data" / "crimes_monthly.csv"
TRAIN_CSV_PATH    = ROOT / "data" / "crimes_train.csv"
VALIDATE_CSV_PATH = ROOT / "data" / "crimes_validate.csv"
FORECAST_CSV_PATH = ROOT / "data" / "crimes_forecast.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "forecaster.joblib"

TRAIN_COLS = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------
print(f"Loading {MONTHLY_PATH} ...")
df_monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])

df_train    = df_monthly.loc[df_monthly["TTV"] == "train",    TRAIN_COLS].copy()
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", TRAIN_COLS].copy()
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", TRAIN_COLS].copy()

print(f"df_train    records: {len(df_train):,}")
print(f"df_validate records: {len(df_validate):,}")
print(f"df_forecast records: {len(df_forecast):,}")

df_train.to_csv(TRAIN_CSV_PATH, index=False)
df_validate.to_csv(VALIDATE_CSV_PATH, index=False)
df_forecast.to_csv(FORECAST_CSV_PATH, index=False)
print(f"Saved {TRAIN_CSV_PATH}")
print(f"Saved {VALIDATE_CSV_PATH}")
print(f"Saved {FORECAST_CSV_PATH}")

print("\n--- 5 random records from df_train ---")
print(df_train.sample(5, random_state=SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Pivot to wide-format panel: rows=date, cols=(ward, primary_type) series
# ---------------------------------------------------------------------------
def series_name(ward, primary_type):
    pt = primary_type.replace(" ", "_").replace("-", "_")
    return f"w{int(ward)}__{pt}"


df_monthly["series"] = df_monthly.apply(
    lambda r: series_name(r["ward"], r["primary_type"]), axis=1,
)
panel = (
    df_monthly
    .pivot_table(index="date", columns="series", values="count_0", aggfunc="first")
    .sort_index()
    .asfreq("MS")
)
print(f"\nPanel shape: {panel.shape[0]} months × {panel.shape[1]} series")

exog = pd.DataFrame({"year": panel.index.year, "month": panel.index.month}, index=panel.index)

fit_mask = panel.index < CUTOFF_DATE
panel_fit = panel.loc[fit_mask]
exog_fit  = exog.loc[fit_mask]
print(f"Fitting on {len(panel_fit)} months of history ({panel_fit.index.min().date()} "
      f"to {panel_fit.index.max().date()})")


# ---------------------------------------------------------------------------
# 3. Fit the ForecasterRecursiveMultiSeries
# ---------------------------------------------------------------------------
regressor = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1,
)
forecaster = ForecasterRecursiveMultiSeries(
    regressor=regressor,
    lags=LAGS,
    encoding="ordinal",
    transformer_series=None,
    transformer_exog=None,
)

print("\nFitting ForecasterRecursiveMultiSeries (XGBoost, 6 lags, year+month exog) ...")
forecaster.fit(series=panel_fit, exog=exog_fit, suppress_warnings=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(forecaster, MODEL_PATH)
print(f"Saved {MODEL_PATH}")
