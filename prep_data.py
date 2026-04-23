"""
prep_data.py

Prepares the Chicago crime data for ML forecasting. Reads data/crimes.csv,
aggregates to monthly counts per (ward, primary_type), zero-pads missing
combinations, fills remaining NaNs, builds lag/lead targets, and tags
train/validate/forecast/incomplete rows. Saves data/crimes_monthly.csv.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load data/crimes.csv into df_filtered.
# ---------------------------------------------------------------------------

df_filtered = pd.read_csv("data/crimes.csv", low_memory=False)
print("df_filtered shape:", df_filtered.shape)


# ---------------------------------------------------------------------------
# Rename date -> timestamp, derive month, keep top-20 primary_type -> df_20.
# ---------------------------------------------------------------------------

df = df_filtered.rename(columns={"date": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["month"] = df["timestamp"].dt.month

top_20_types = df["primary_type"].value_counts().head(20).index.tolist()
df_20 = df[df["primary_type"].isin(top_20_types)].copy()

print(f"\ndf_20 has {len(df_20):,} records")

print("\nprimary_type counts in df_20:")
print(df_20["primary_type"].value_counts())

print("\n1 random record in df_20 (all columns):")
print(df_20.sample(n=1, random_state=None).to_string())

print("\ncolumn types in df_20:")
print(df_20.dtypes)

print(f"\ndf_20 timestamp min: {df_20['timestamp'].min()}")
print(f"df_20 timestamp max: {df_20['timestamp'].max()}")


# ---------------------------------------------------------------------------
# Group by (year, month, ward, primary_type) -> df_avg.
# ---------------------------------------------------------------------------

df_avg = (
    df_20.dropna(subset=["ward"])
    .groupby(["year", "month", "ward", "primary_type"], as_index=False)
    .agg(
        arrest=("arrest", "mean"),
        domestic=("domestic", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        count_0=("id", "count"),
    )
)

# cast ward to int, add day=1, reorder so count_0 is last, sort
df_avg["ward"] = df_avg["ward"].astype(int)
df_avg["day"] = 1
ordered_cols = [
    "year", "month", "day", "ward", "primary_type",
    "arrest", "domestic", "latitude", "longitude", "count_0",
]
df_avg = df_avg[ordered_cols].sort_values(
    ["year", "month", "ward", "primary_type"]
).reset_index(drop=True)

print(f"\ndf_avg has {len(df_avg):,} records")


# ---------------------------------------------------------------------------
# df_date = df_avg + derived date column.
# ---------------------------------------------------------------------------

df_date = df_avg.copy()
df_date["date"] = pd.to_datetime(df_date[["year", "month", "day"]])

# Which ward has greatest sum(count_0) among THEFT?
theft = df_date[df_date["primary_type"] == "THEFT"]
theft_sum_by_ward = theft.groupby("ward")["count_0"].sum().sort_values(ascending=False)
top_theft_ward = int(theft_sum_by_ward.index[0])
print(f"\nward with greatest sum(count_0) among THEFT: {top_theft_ward} "
      f"(sum={int(theft_sum_by_ward.iloc[0]):,})")

print("\nAll THEFT records in df_date where ward=42:")
print(df_date[(df_date["primary_type"] == "THEFT") & (df_date["ward"] == 42)].to_string())

print("\nAll ARSON records in df_date where ward=42:")
print(df_date[(df_date["primary_type"] == "ARSON") & (df_date["ward"] == 42)].to_string())


# ---------------------------------------------------------------------------
# Zero-pad missing (date, ward, primary_type) combinations -> df_pad.
# ---------------------------------------------------------------------------

all_dates = sorted(df_date["date"].unique())
all_wards = sorted(df_date["ward"].unique())
all_types = sorted(df_date["primary_type"].unique())

full_index = pd.MultiIndex.from_product(
    [all_dates, all_wards, all_types],
    names=["date", "ward", "primary_type"],
).to_frame(index=False)

df_pad = full_index.merge(df_date, on=["date", "ward", "primary_type"], how="left")

# fill count_0 with 0 where missing; also fill year/month from the date column
df_pad["count_0"] = df_pad["count_0"].fillna(0).astype(int)
df_pad["year"] = df_pad["date"].dt.year
df_pad["month"] = df_pad["date"].dt.month
df_pad["day"] = 1

# reorder columns to match df_avg (plus date at the end)
df_pad = df_pad[
    [
        "year", "month", "day", "ward", "primary_type",
        "arrest", "domestic", "latitude", "longitude", "count_0", "date",
    ]
].sort_values(["date", "ward", "primary_type"]).reset_index(drop=True)

print(f"\ndf_pad has {len(df_pad):,} records")

print("\nAll ARSON records in df_pad where ward=42:")
print(df_pad[(df_pad["primary_type"] == "ARSON") & (df_pad["ward"] == 42)].to_string())


# ---------------------------------------------------------------------------
# Replace NaN in each column with random picks from that column's non-NaN
# values -> df_nan.
# ---------------------------------------------------------------------------

df_nan = df_pad.copy()
rng = np.random.default_rng(seed=42)

for col in df_nan.columns:
    mask = df_nan[col].isna()
    n_missing = int(mask.sum())
    if n_missing == 0:
        continue
    non_na = df_nan.loc[~mask, col].to_numpy()
    if len(non_na) == 0:
        continue
    picks = rng.choice(non_na, size=n_missing, replace=True)
    df_nan.loc[mask, col] = picks

print("\nAll ARSON records in df_nan where ward=22:")
print(df_nan[(df_nan["primary_type"] == "ARSON") & (df_nan["ward"] == 22)].to_string())

print("\n5 random records in df_nan:")
print(df_nan.sample(n=5, random_state=None).to_string())


# ---------------------------------------------------------------------------
# Build lag/lead targets -> df_target.
# Group by (ward, primary_type), order by date; shift count_0 to form
# count_previous (1 month back) and count_1..count_4 (1..4 months forward).
# ---------------------------------------------------------------------------

df_target = df_nan.sort_values(["ward", "primary_type", "date"]).reset_index(drop=True)

grp = df_target.groupby(["ward", "primary_type"])["count_0"]
df_target["count_previous"] = grp.shift(1)   # 1 month earlier
df_target["count_1"] = grp.shift(-1)          # 1 month ahead
df_target["count_2"] = grp.shift(-2)
df_target["count_3"] = grp.shift(-3)
df_target["count_4"] = grp.shift(-4)

df_target["delta_count"] = df_target["count_0"] - df_target["count_previous"]

# reorder: date first, then the rest, with the target-shift block last
target_cols = ["delta_count", "count_0", "count_1", "count_2", "count_3", "count_4"]
other_cols = [c for c in df_target.columns if c not in target_cols + ["date"]]
df_target = df_target[["date"] + other_cols + target_cols]

print("\nAll THEFT records in df_target where ward=27:")
print(df_target[(df_target["primary_type"] == "THEFT") & (df_target["ward"] == 27)].to_string())


# ---------------------------------------------------------------------------
# Partition into train / validate / forecast / incomplete -> df_ttv.
# ---------------------------------------------------------------------------

df_ttv = df_target.copy()
cutoff = pd.Timestamp("2025-01-01")
df_ttv["TTV"] = np.where(df_ttv["date"] < cutoff, "train", "validate")

# top 2 dates -> 'forecast'
unique_dates_desc = sorted(df_ttv["date"].unique(), reverse=True)
last_few_dates = unique_dates_desc[:2]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "forecast"

# top 1 date -> 'incomplete'
last_few_dates = unique_dates_desc[:1]
df_ttv.loc[df_ttv["date"].isin(last_few_dates), "TTV"] = "incomplete"


# ---------------------------------------------------------------------------
# Drop columns -> df_monthly; drop rows with delta_count=NaN.
# ---------------------------------------------------------------------------

df_monthly = df_ttv.drop(columns=["arrest", "domestic", "count_previous"])
df_monthly = df_monthly.dropna(subset=["delta_count"]).reset_index(drop=True)

# pretty-print ALL records where primary_type=THEFT and ward=22
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
print("\nAll THEFT records in df_monthly where ward=22:")
print(df_monthly[(df_monthly["primary_type"] == "THEFT") & (df_monthly["ward"] == 22)].to_string())
pd.reset_option("display.max_rows")
pd.reset_option("display.width")


# ---------------------------------------------------------------------------
# Save df_monthly.
# ---------------------------------------------------------------------------

out_path = "data/crimes_monthly.csv"
df_monthly.to_csv(out_path, index=False)
print(f"\nsaved df_monthly to {out_path} ({len(df_monthly):,} records)")
