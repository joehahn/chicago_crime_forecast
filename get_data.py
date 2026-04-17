#!/usr/bin/env python3
"""Download the Chicago crimes dataset (past 4 years) and do light filtering & cleanup.

Source: Chicago Data Portal (Socrata API), dataset ID ijzp-q8t2.
Output: data/crimes.csv
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sodapy import Socrata

DATASET_ID = "ijzp-q8t2"
DOMAIN = "data.cityofchicago.org"
YEARS_BACK = 4
ROW_LIMIT = 2_000_000
EXCLUDED_PRIMARY_TYPES = [
    "CRIMINAL SEXUAL ASSAULT",
    "OFFENSE INVOLVING CHILDREN",
    "SEX OFFENSE",
    "PROSTITUTION",
]
OUTPUT_PATH = Path(__file__).parent / "data" / "crimes.csv"


# 1. Download past 4 years of data. Always re-download from source — no local cache.
cutoff = (datetime.now() - timedelta(days=YEARS_BACK * 365)).strftime("%Y-%m-%dT%H:%M:%S")
print(f"Downloading {DATASET_ID} from {DOMAIN} (records since {cutoff})...")
client = Socrata(DOMAIN, None)
records = client.get(DATASET_ID, where=f"date > '{cutoff}'", limit=ROW_LIMIT)
df_raw = pd.DataFrame.from_records(records)
print(f"Records in df_raw:      {len(df_raw):,}")

# 2. Mitigate problematic carriage returns and newlines in text columns.
df_clean = df_raw.copy()
for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = (
        df_clean[col]
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
    )
print(f"Records in df_clean:    {len(df_clean):,}")

# 3. Show primary_type distribution.
print("\nprimary_type counts in df_clean:")
print(df_clean["primary_type"].value_counts().to_string())

# 4. Filter out excluded primary_type values.
df_filtered = df_clean[~df_clean["primary_type"].isin(EXCLUDED_PRIMARY_TYPES)].copy()
print(f"\nRecords in df_filtered: {len(df_filtered):,}")

# 5. Show 1 random record.
print("\nRandom record from df_filtered:")
print(df_filtered.sample(1).to_string())

# 6. Report date range.
print(f"\nMin date: {df_filtered['date'].min()}")
print(f"Max date: {df_filtered['date'].max()}")

# 7. Save to CSV.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_filtered.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved df_filtered to {OUTPUT_PATH}")
