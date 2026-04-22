#!/usr/bin/env python3
# get_data.py
# Download Chicago crimes dataset from the Socrata API and do light filtering & cleanup.

import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta

# Download past 4 years of crime data from Chicago Data Portal
print("Downloading Chicago crime data...")
cutoff = (datetime.now() - timedelta(days=4 * 365)).strftime("%Y-%m-%dT%H:%M:%S")
client = Socrata("data.cityofchicago.org", None)
results = client.get("ijzp-q8t2", where=f"date > '{cutoff}'", limit=2_000_000)
df_raw = pd.DataFrame.from_records(results)
print(f"Records in df_raw: {len(df_raw):,}")

# Inspect text columns and mitigate carriage returns
print("\nCleaning carriage returns from text columns...")
text_cols = df_raw.select_dtypes(include="object").columns
df_clean = df_raw.copy()
for col in text_cols:
    df_clean[col] = (
        df_clean[col]
        .astype(str)
        .str.replace(r"\r", " ", regex=True)
        .str.replace(r"\n", " ", regex=True)
    )
print(f"Records in df_clean: {len(df_clean):,}")

# Count by primary_type
print("\nprimary_type counts:")
print(df_clean["primary_type"].value_counts().to_string())

# Filter out sensitive crime types
exclude = [
    "CRIMINAL SEXUAL ASSAULT",
    "OFFENSE INVOLVING CHILDREN",
    "SEX OFFENSE",
    "PROSTITUTION",
]
df_filtered = df_clean[~df_clean["primary_type"].isin(exclude)].copy()
print(f"\nRecords in df_filtered: {len(df_filtered):,}")

# Display 1 random record
print("\nRandom record from df_filtered:")
print(df_filtered.sample(1).to_string())

# Min and max date
print(f"\nMin date: {df_filtered['date'].min()}")
print(f"Max date: {df_filtered['date'].max()}")

# Save to CSV
output_path = "data/crimes.csv"
df_filtered.to_csv(output_path, index=False)
print(f"\nSaved df_filtered to {output_path}")
