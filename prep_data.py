#!/usr/bin/env python3
# prep_data.py
# Prepare the Chicago crime data for ML training: load, clean, aggregate, and feature-engineer.

import numpy as np
import pandas as pd

# Load the cleaned & filtered crimes dataset saved by get_data.py
df_filtered = pd.read_csv("data/crimes.csv", low_memory=False)
print(f"df_filtered shape: {df_filtered.shape}")

# Rename date -> timestamp, derive month-of-year, and keep only the top-20 primary_type values
df_filtered = df_filtered.rename(columns={"date": "timestamp"})
df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
df_filtered["month"] = df_filtered["timestamp"].dt.month
top20 = df_filtered["primary_type"].value_counts().head(20).index
df_20 = df_filtered[df_filtered["primary_type"].isin(top20)].copy()
print(f"Records in df_20: {len(df_20):,}")

# Show count of each primary_type remaining in df_20
print("\nprimary_type counts in df_20:")
print(df_20["primary_type"].value_counts().to_string())

# Display 1 random record with all columns visible
print("\nRandom record from df_20:")
with pd.option_context("display.max_columns", None, "display.width", None):
    print(df_20.sample(1).to_string())

# Show column dtypes
print("\ndf_20 dtypes:")
print(df_20.dtypes.to_string())

# Report timestamp range in df_20
print(f"\nMin timestamp: {df_20['timestamp'].min()}")
print(f"Max timestamp: {df_20['timestamp'].max()}")

# Aggregate by (year, month, ward, primary_type): means of a few fields + row count
df_avg = (
    df_20.groupby(["year", "month", "ward", "primary_type"], as_index=False)
    .agg(
        arrest=("arrest", "mean"),
        domestic=("domestic", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        count_0=("id", "count"),
    )
)
df_avg["ward"] = df_avg["ward"].astype(int)
df_avg["day"] = 1
# Put count_0 last; everything else keeps its natural order
cols = [c for c in df_avg.columns if c != "count_0"] + ["count_0"]
df_avg = df_avg[cols]
df_avg = df_avg.sort_values(["year", "month", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_avg: {len(df_avg):,}")

# Derive a proper datetime column from the (year, month, day) integer triple
df_date = df_avg
df_date["date"] = pd.to_datetime(df_date[["year", "month", "day"]])
print(f"df_date columns: {list(df_date.columns)}")
print(f"Min date: {df_date['date'].min()}   Max date: {df_date['date'].max()}")

# Build a dense (date, ward, primary_type) grid and left-join df_date onto it,
# so every (month, ward, crime type) combination is represented (zero-pad missing ones).
all_dates = pd.date_range(df_date["date"].min(), df_date["date"].max(), freq="MS")
all_wards = sorted(df_date["ward"].unique())
all_types = sorted(df_date["primary_type"].unique())
grid = pd.MultiIndex.from_product(
    [all_dates, all_wards, all_types],
    names=["date", "ward", "primary_type"],
).to_frame(index=False)

df_pad = grid.merge(df_date, on=["date", "ward", "primary_type"], how="left")
df_pad["count_0"] = df_pad["count_0"].fillna(0).astype(int)
df_pad["year"] = df_pad["date"].dt.year
df_pad["month"] = df_pad["date"].dt.month
df_pad["day"] = 1
df_pad = df_pad.sort_values(["date", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_pad: {len(df_pad):,}")

# For each column, replace NaN values with random draws from that column's non-NaN values
rng = np.random.default_rng(42)
df_nan = df_pad.copy()
print("\nNaN counts per column before imputation:")
print(df_nan.isna().sum()[df_nan.isna().sum() > 0].to_string() or "(none)")
for col in df_nan.columns:
    mask = df_nan[col].isna()
    if mask.any():
        pool = df_nan.loc[~mask, col].to_numpy()
        df_nan.loc[mask, col] = rng.choice(pool, size=mask.sum(), replace=True)
print("\nNaN counts per column after imputation:")
post = df_nan.isna().sum()
print(post[post > 0].to_string() or "(none)")

# Build lag/lead count features within each (ward, primary_type) time series, sorted by date.
# NaNs appear at each series' edges where the shift falls outside the observed range.
df_target = df_nan.sort_values(["ward", "primary_type", "date"]).reset_index(drop=True)
grp = df_target.groupby(["ward", "primary_type"], sort=False)["count_0"]
df_target["count_previous"] = grp.shift(1)
df_target["count_1"] = grp.shift(-1)
df_target["count_2"] = grp.shift(-2)
df_target["count_3"] = grp.shift(-3)
df_target["count_4"] = grp.shift(-4)
df_target["delta_count"] = df_target["count_0"] - df_target["count_previous"]

# Move date to the front and the delta/count block to the back; everything else keeps its order.
last = ["delta_count", "count_0", "count_1", "count_2", "count_3", "count_4"]
middle = [c for c in df_target.columns if c not in (["date"] + last)]
df_target = df_target[["date"] + middle + last]
print(f"\ndf_target shape: {df_target.shape}")
print(f"df_target columns: {list(df_target.columns)}")

# Randomly assign each row to 'train' (~2/3) or 'test' (~1/3) via a uniform [0,1) draw
df_ttv = df_target.copy()
df_ttv["ran_num"] = np.random.default_rng(123).random(len(df_ttv))
df_ttv["TTV"] = np.where(df_ttv["ran_num"] <= 0.667, "train", "test")
# Override: everything from 2025-01-01 onward is the out-of-time validation window
df_ttv.loc[df_ttv["date"] >= "2025-01-01", "TTV"] = "validate"
# Flag the latest 2 months as the live forecast horizon (where future-lag targets are mostly NaN)
last_few_dates = sorted(df_ttv["date"].unique())[-2:]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "forecast"
print(f"last_few_dates (forecast): {[pd.Timestamp(d).date() for d in last_few_dates]}")
# The very last month's source data is only partial — tag it separately so it can be excluded downstream
last_few_dates = sorted(df_ttv["date"].unique())[-1:]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "incomplete"
print(f"last_few_dates (incomplete): {[pd.Timestamp(d).date() for d in last_few_dates]}")
print(f"\nTTV split counts:\n{df_ttv['TTV'].value_counts().to_string()}")

# Drop bookkeeping & mean-of-bool columns that won't be used downstream
df_monthly = df_ttv.drop(columns=["arrest", "domestic", "count_previous", "ran_num"])
print(f"\ndf_monthly shape: {df_monthly.shape}")
print(f"df_monthly columns: {list(df_monthly.columns)}")

# Drop the first month of each series (delta_count undefined without a prior month)
before = len(df_monthly)
df_monthly = df_monthly.dropna(subset=["delta_count"]).reset_index(drop=True)
print(f"\nDropped {before - len(df_monthly):,} rows with NaN delta_count; df_monthly now: {df_monthly.shape}")

# Pretty-print all records for (primary_type=THEFT, ward=22) as a sanity check
sub = df_monthly[(df_monthly["primary_type"] == "THEFT") & (df_monthly["ward"] == 22)]
print(f"\nAll df_monthly rows for primary_type=THEFT, ward=22 ({len(sub)} rows):")
with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.width", None,
    "display.expand_frame_repr", False,
):
    print(sub.to_string(index=False))

# Persist the prepared monthly panel for downstream model training / validation
output_path = "data/crimes_monthly.csv"
df_monthly.to_csv(output_path, index=False)
print(f"\nSaved df_monthly to {output_path}")
