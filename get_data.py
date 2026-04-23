"""
get_data.py

Downloads the Chicago crimes dataset from the Chicago Data Portal (Socrata API)
and performs light filtering to make it business-friendly. Saves the result
to data/crimes.csv.

Source: https://www.chicago.gov/city/en/dataset/crime.html
Socrata dataset ID: ijzp-q8t2
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Download the latest 4 years of Chicago crime data via the Socrata API.
# Dataset ID = ijzp-q8t2. Always re-download; do not use local cache.
# ---------------------------------------------------------------------------

# compute cutoff date = today minus 4 years, formatted for SoQL $where clause
cutoff_date = (datetime.utcnow() - timedelta(days=4 * 365)).strftime("%Y-%m-%dT00:00:00")

# Socrata endpoint for the Chicago Crimes dataset
endpoint = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

# page through the API in chunks (Socrata caps a single request's page size)
page_size = 50000
offset = 0
all_rows = []

print(f"Downloading Chicago crime records since {cutoff_date} ...")
while True:
    params = {
        "$where": f"date >= '{cutoff_date}'",
        "$limit": page_size,
        "$offset": offset,
        "$order": "date",
    }
    response = requests.get(endpoint, params=params, timeout=120)
    response.raise_for_status()
    rows = response.json()
    if not rows:
        break
    all_rows.extend(rows)
    print(f"  fetched {len(rows):,} records (running total: {len(all_rows):,})")
    if len(rows) < page_size:
        break
    offset += page_size

# assemble the full raw dataframe
df_raw = pd.DataFrame(all_rows)
print(f"\ndf_raw has {df_raw.shape[1]} columns and {df_raw.shape[0]:,} records")


# ---------------------------------------------------------------------------
# Clean problematic carriage returns / newlines from text columns.
# ---------------------------------------------------------------------------

df_clean = df_raw.copy()

# find all object (string) columns
text_cols = df_clean.select_dtypes(include=["object"]).columns
for col in text_cols:
    # replace any embedded CR/LF characters with a single space
    df_clean[col] = (
        df_clean[col]
        .astype(str)
        .str.replace("\r\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
    )

print(f"\ndf_clean has {df_clean.shape[0]:,} records")


# ---------------------------------------------------------------------------
# Display a count of primary_type values in df_clean.
# ---------------------------------------------------------------------------

print("\nprimary_type counts in df_clean:")
print(df_clean["primary_type"].value_counts())


# ---------------------------------------------------------------------------
# Filter out sensitive / sex-related categories.
# ---------------------------------------------------------------------------

excluded_types = [
    "CRIMINAL SEXUAL ASSAULT",
    "OFFENSE INVOLVING CHILDREN",
    "SEX OFFENSE",
    "PROSTITUTION",
]

df_filtered = df_clean[~df_clean["primary_type"].isin(excluded_types)].copy()
print(f"\ndf_filtered has {df_filtered.shape[0]:,} records")


# ---------------------------------------------------------------------------
# Display 1 random record from df_filtered.
# ---------------------------------------------------------------------------

print("\n1 random record from df_filtered:")
print(df_filtered.sample(n=1, random_state=None).to_string())


# ---------------------------------------------------------------------------
# Report min and max of the date column.
# ---------------------------------------------------------------------------

date_min = df_filtered["date"].min()
date_max = df_filtered["date"].max()
print(f"\ndate min: {date_min}")
print(f"date max: {date_max}")


# ---------------------------------------------------------------------------
# Save df_filtered to data/crimes.csv.
# ---------------------------------------------------------------------------

os.makedirs("data", exist_ok=True)
out_path = "data/crimes.csv"
df_filtered.to_csv(out_path, index=False)
print(f"\nsaved df_filtered to {out_path}")
