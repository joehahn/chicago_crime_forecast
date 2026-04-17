#!/usr/bin/env python3
"""Prepare Chicago crime data for ML training.

Reads  : data/crimes.csv      (produced by get_data.py)
Writes : data/crimes_monthly.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOP_N_PRIMARY_TYPES = 20
VALIDATE_FROM = "2025-01-01"
TRAIN_THRESHOLD = 0.667

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "data" / "crimes.csv"
OUTPUT_PATH = ROOT / "data" / "crimes_monthly.csv"

rng = np.random.default_rng(SEED)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# 1. Load.
print(f"Loading {INPUT_PATH} ...")
df_filtered = pd.read_csv(INPUT_PATH)
print(f"df_filtered shape: {df_filtered.shape}")

# 2. Profile.
print("\n--- Profile df_filtered ---")
print(df_filtered.dtypes)
print()
print(df_filtered.describe(include="all"))

# 3. Rename date→timestamp, derive month & year, keep top-20 primary_type.
df_filtered = df_filtered.rename(columns={"date": "timestamp"})
df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
df_filtered["month"] = df_filtered["timestamp"].dt.month
df_filtered["year"] = df_filtered["timestamp"].dt.year

top_types = df_filtered["primary_type"].value_counts().nlargest(TOP_N_PRIMARY_TYPES).index
df_20 = df_filtered[df_filtered["primary_type"].isin(top_types)].copy()
print(f"\nRecords in df_20: {len(df_20):,}")

print("\n--- primary_type counts in df_20 ---")
print(df_20["primary_type"].value_counts().to_string())

print("\n--- 1 random record in df_20 ---")
print(df_20.sample(1, random_state=SEED).to_string())

print("\n--- Column types in df_20 ---")
print(df_20.dtypes)

print(f"\nMin date in df_20: {df_20['timestamp'].min()}")
print(f"Max date in df_20: {df_20['timestamp'].max()}")

# 4. Aggregate → df_avg.
df_avg = (
    df_20.groupby(["year", "month", "ward", "primary_type"])
    .agg(
        mean_arrest=("arrest", "mean"),
        mean_domestic=("domestic", "mean"),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean"),
        count_0=("id", "count"),
    )
    .reset_index()
)
df_avg.columns = [c.replace("mean_", "") for c in df_avg.columns]
df_avg["ward"] = df_avg["ward"].astype(int)
df_avg["day"] = 1
df_avg = df_avg[[c for c in df_avg.columns if c != "count_0"] + ["count_0"]]
df_avg = df_avg.sort_values(["year", "month", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_avg: {len(df_avg):,}")

# 5. df_date = df_avg with a proper date column.
df_date = df_avg.copy()
df_date["date"] = pd.to_datetime(df_date[["year", "month", "day"]])

theft_by_ward = df_date.loc[df_date["primary_type"] == "THEFT"].groupby("ward")["count_0"].sum()
print(f"\nWard with greatest sum(count_0) for THEFT: {theft_by_ward.idxmax()}")

print("\n--- df_date: primary_type=THEFT, ward=42 ---")
print(df_date[(df_date["primary_type"] == "THEFT") & (df_date["ward"] == 42)].to_string())

print("\n--- df_date: primary_type=ARSON, ward=42 ---")
print(df_date[(df_date["primary_type"] == "ARSON") & (df_date["ward"] == 42)].to_string())

# 6. Zero-pad missing (date, ward, primary_type) combinations → df_pad.
full_index = pd.MultiIndex.from_product(
    [sorted(df_date["date"].unique()),
     sorted(df_date["ward"].unique()),
     sorted(df_date["primary_type"].unique())],
    names=["date", "ward", "primary_type"],
)
df_pad = (
    pd.DataFrame(index=full_index)
    .reset_index()
    .merge(df_date, on=["date", "ward", "primary_type"], how="left")
)
df_pad["count_0"] = df_pad["count_0"].fillna(0).astype(int)
df_pad["year"] = df_pad["date"].dt.year
df_pad["month"] = df_pad["date"].dt.month
df_pad["day"] = df_pad["date"].dt.day
df_pad = df_pad.sort_values(["date", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_pad: {len(df_pad):,}")

print("\n--- df_pad: primary_type=ARSON, ward=42 ---")
print(df_pad[(df_pad["primary_type"] == "ARSON") & (df_pad["ward"] == 42)].to_string())

# 7. Impute NaN with random samples from the same column → df_nan.
df_nan = df_pad.copy()
for col in df_nan.columns:
    nan_mask = df_nan[col].isna()
    if nan_mask.any():
        non_nan_vals = df_nan.loc[~nan_mask, col].values
        df_nan.loc[nan_mask, col] = rng.choice(non_nan_vals, size=nan_mask.sum())

print("\n--- df_nan: primary_type=ARSON, ward=22 ---")
print(df_nan[(df_nan["primary_type"] == "ARSON") & (df_nan["ward"] == 22)].to_string())

print("\n--- 5 random records in df_nan ---")
print(df_nan.sample(5, random_state=SEED).to_string())

# 8. Shift features → df_target.
def add_shifts(group):
    group = group.sort_values("date")
    group["count_previous"] = group["count_0"].shift(1)
    group["count_1"] = group["count_0"].shift(-1)
    group["count_2"] = group["count_0"].shift(-2)
    group["count_3"] = group["count_0"].shift(-3)
    group["count_4"] = group["count_0"].shift(-4)
    return group


df_target = df_nan.groupby(["ward", "primary_type"], group_keys=False).apply(add_shifts)
df_target["delta_count"] = df_target["count_0"] - df_target["count_previous"]

shift_cols = ["delta_count", "count_0", "count_1", "count_2", "count_3", "count_4"]
other_cols = [c for c in df_target.columns if c != "date" and c not in shift_cols]
df_target = df_target[["date"] + other_cols + shift_cols]
df_target = df_target.sort_values(["date", "ward", "primary_type"]).reset_index(drop=True)

print("\n--- df_target: primary_type=THEFT, ward=27 ---")
print(df_target[(df_target["primary_type"] == "THEFT") & (df_target["ward"] == 27)].to_string())

# 9. Assign TTV splits → df_ttv.
df_ttv = df_target.copy()
df_ttv["ran_num"] = rng.uniform(0.0, 1.0, size=len(df_ttv))
df_ttv["TTV"] = np.where(df_ttv["ran_num"] <= TRAIN_THRESHOLD, "train", "test")
df_ttv.loc[df_ttv["date"] >= VALIDATE_FROM, "TTV"] = "validate"

sorted_dates = sorted(df_ttv["date"].unique())
df_ttv.loc[df_ttv["date"].isin(sorted_dates[-2:]), "TTV"] = "forecast"
df_ttv.loc[df_ttv["date"] == sorted_dates[-1], "TTV"] = "incomplete"

# 10. Drop columns → df_monthly, drop delta_count NaNs.
df_monthly = df_ttv.drop(columns=["arrest", "domestic", "count_previous", "ran_num"])
df_monthly = df_monthly.dropna(subset=["delta_count"]).reset_index(drop=True)

print("\n--- df_monthly: primary_type=THEFT, ward=22 ---")
pd.set_option("display.max_rows", None)
print(df_monthly[(df_monthly["primary_type"] == "THEFT") & (df_monthly["ward"] == 22)].to_string())
pd.reset_option("display.max_rows")

# 11. Save.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_monthly.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved df_monthly to {OUTPUT_PATH} ({len(df_monthly):,} records)")
