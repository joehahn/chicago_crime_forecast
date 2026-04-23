#!/usr/bin/env python3
# prep_data.py
# Prepare the Chicago crime data for ML training: load, filter, aggregate, pad, impute, feature-engineer.

import numpy as np
import pandas as pd

# Pretty-print helper: show every column / row without pandas truncation
PP = dict(
    display_max_rows=None,
    display_max_columns=None,
    display_width=None,
    display_expand_frame_repr=False,
)
def _show(df):
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        print(df.to_string(index=False))


# --- Load the cleaned crimes dataset -------------------------------------------------
df_filtered = pd.read_csv("data/crimes.csv", low_memory=False)
print(f"df_filtered shape: {df_filtered.shape}")


# --- Rename date->timestamp, derive month-of-year, keep top-20 primary_types ---------
df_filtered = df_filtered.rename(columns={"date": "timestamp"})
df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
df_filtered["month"] = df_filtered["timestamp"].dt.month
top20 = df_filtered["primary_type"].value_counts().head(20).index
df_20 = df_filtered[df_filtered["primary_type"].isin(top20)].copy()
print(f"\nRecords in df_20: {len(df_20):,}")

print("\nprimary_type counts in df_20:")
print(df_20["primary_type"].value_counts().to_string())

print("\nRandom record from df_20 (all columns):")
with pd.option_context("display.max_columns", None, "display.width", None):
    print(df_20.sample(1, random_state=0).to_string())

print("\ndf_20 dtypes:")
print(df_20.dtypes.to_string())

print(f"\nMin timestamp in df_20: {df_20['timestamp'].min()}")
print(f"Max timestamp in df_20: {df_20['timestamp'].max()}")


# --- Aggregate by (year, month, ward, primary_type) ----------------------------------
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
# Put count_0 last; everything else keeps natural order
cols = [c for c in df_avg.columns if c != "count_0"] + ["count_0"]
df_avg = df_avg[cols]
df_avg = df_avg.sort_values(["year", "month", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_avg: {len(df_avg):,}")


# --- Derive a proper date column from (year, month, day) -----------------------------
df_date = df_avg.copy()
df_date["date"] = pd.to_datetime(df_date[["year", "month", "day"]])

# Which ward has the greatest sum(count_0) among THEFT records?
theft_by_ward = (
    df_date[df_date["primary_type"] == "THEFT"]
    .groupby("ward", as_index=False)["count_0"].sum()
    .sort_values("count_0", ascending=False)
)
top_theft_ward = int(theft_by_ward.iloc[0]["ward"])
top_theft_count = int(theft_by_ward.iloc[0]["count_0"])
print(f"\nWard with greatest sum(count_0) for THEFT: ward={top_theft_ward} (count_0 sum={top_theft_count:,})")

print("\nAll df_date records for primary_type=THEFT, ward=42:")
_show(df_date[(df_date["primary_type"] == "THEFT") & (df_date["ward"] == 42)])

print("\nAll df_date records for primary_type=ARSON, ward=42:")
_show(df_date[(df_date["primary_type"] == "ARSON") & (df_date["ward"] == 42)])


# --- Zero-pad missing (date, ward, primary_type) combinations ------------------------
all_dates = pd.date_range(df_date["date"].min(), df_date["date"].max(), freq="MS")
all_wards = sorted(df_date["ward"].unique())
all_types = sorted(df_date["primary_type"].unique())
grid = pd.MultiIndex.from_product(
    [all_dates, all_wards, all_types],
    names=["date", "ward", "primary_type"],
).to_frame(index=False)

df_pad = grid.merge(df_date, on=["date", "ward", "primary_type"], how="left")
df_pad["count_0"] = df_pad["count_0"].fillna(0).astype(int)
# Fill year/month/day from the padded date so every row has them
df_pad["year"] = df_pad["date"].dt.year
df_pad["month"] = df_pad["date"].dt.month
df_pad["day"] = 1
df_pad = df_pad.sort_values(["date", "ward", "primary_type"]).reset_index(drop=True)
print(f"\nRecords in df_pad: {len(df_pad):,}")

print("\nAll df_pad records for primary_type=ARSON, ward=42:")
_show(df_pad[(df_pad["primary_type"] == "ARSON") & (df_pad["ward"] == 42)])


# --- Replace NaN in each column with random draws from that column's non-NaN values --
rng = np.random.default_rng(42)
df_nan = df_pad.copy()
print("\nNaN counts per column before imputation:")
pre = df_nan.isna().sum()
print(pre[pre > 0].to_string() or "(none)")
for col in df_nan.columns:
    mask = df_nan[col].isna()
    if mask.any():
        pool = df_nan.loc[~mask, col].to_numpy()
        df_nan.loc[mask, col] = rng.choice(pool, size=mask.sum(), replace=True)
print("\nNaN counts per column after imputation:")
post = df_nan.isna().sum()
print(post[post > 0].to_string() or "(none)")

print("\nAll df_nan records for primary_type=ARSON, ward=22:")
_show(df_nan[(df_nan["primary_type"] == "ARSON") & (df_nan["ward"] == 22)])

print("\n5 random records from df_nan:")
_show(df_nan.sample(5, random_state=7))


# --- Lag/lead count features within each (ward, primary_type) series, sorted by date -
df_target = df_nan.sort_values(["ward", "primary_type", "date"]).reset_index(drop=True)
grp = df_target.groupby(["ward", "primary_type"], sort=False)["count_0"]
df_target["count_previous"] = grp.shift(1)
df_target["count_1"] = grp.shift(-1)
df_target["count_2"] = grp.shift(-2)
df_target["count_3"] = grp.shift(-3)
df_target["count_4"] = grp.shift(-4)
df_target["delta_count"] = df_target["count_0"] - df_target["count_previous"]

# Move `date` to the front and `delta_count, count_0..count_4` to the back
last = ["delta_count", "count_0", "count_1", "count_2", "count_3", "count_4"]
middle = [c for c in df_target.columns if c not in (["date"] + last)]
df_target = df_target[["date"] + middle + last]
print(f"\ndf_target shape: {df_target.shape}")
print(f"df_target columns: {list(df_target.columns)}")

print("\nAll df_target records for primary_type=THEFT, ward=27:")
_show(df_target[(df_target["primary_type"] == "THEFT") & (df_target["ward"] == 27)])


# --- TTV (train/validate/forecast/incomplete) split by date --------------------------
df_ttv = df_target.copy()
df_ttv["TTV"] = np.where(df_ttv["date"] < "2025-01-01", "train", "validate")
# The last 2 months are the live forecast horizon (future lag targets are mostly NaN)
last_few_dates = sorted(df_ttv["date"].unique())[-2:]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "forecast"
print(f"\nlast_few_dates (forecast): {[pd.Timestamp(d).date() for d in last_few_dates]}")
# The very last month's data is only partial — tag it separately so it can be excluded downstream
last_few_dates = sorted(df_ttv["date"].unique())[-1:]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "incomplete"
print(f"last_few_dates (incomplete): {[pd.Timestamp(d).date() for d in last_few_dates]}")
print(f"\nTTV split counts:\n{df_ttv['TTV'].value_counts().to_string()}")

# Drop bookkeeping / mean-of-bool columns that won't be used downstream
df_monthly = df_ttv.drop(columns=["arrest", "domestic", "count_previous"])

# Drop rows where delta_count is NaN (the first month of each series has no predecessor)
before = len(df_monthly)
df_monthly = df_monthly.dropna(subset=["delta_count"]).reset_index(drop=True)
print(f"\nDropped {before - len(df_monthly):,} rows with NaN delta_count; df_monthly now: {df_monthly.shape}")

print("\nAll df_monthly records for primary_type=THEFT, ward=22:")
_show(df_monthly[(df_monthly["primary_type"] == "THEFT") & (df_monthly["ward"] == 22)])


# --- Persist the prepared monthly panel ----------------------------------------------
output_path = "data/crimes_monthly.csv"
df_monthly.to_csv(output_path, index=False)
print(f"\nSaved df_monthly to {output_path}")
