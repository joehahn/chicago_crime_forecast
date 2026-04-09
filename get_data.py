#!/usr/bin/env python3
"""Load, clean, filter, and profile Chicago crime data.
Download first with: python download_crimes.py
"""

import pandas as pd

# ── 1. Read raw data ──────────────────────────────────────────────────────────
df_raw = pd.read_csv('data/crimes.csv', low_memory=False)
print(f"df_raw records: {len(df_raw):,}")

# ── 2. Mitigate carriage returns in text columns ──────────────────────────────
text_cols = df_raw.select_dtypes(include='object').columns.tolist()
df_clean = df_raw.copy()
for col in text_cols:
    df_clean[col] = df_clean[col].str.replace(r'\r\n|\r|\n', ' ', regex=True)
print(f"df_clean records: {len(df_clean):,}")

# ── 3. primary_type counts ────────────────────────────────────────────────────
print("\nprimary_type counts in df_clean:")
print(df_clean['primary_type'].value_counts().to_string())

# ── 4. Filter sensitive categories ────────────────────────────────────────────
exclude = ['CRIMINAL SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'SEX OFFENSE', 'PROSTITUTION']
df_filtered = df_clean[~df_clean['primary_type'].isin(exclude)].copy()
df_filtered.to_csv('data/crimes_filtered.csv', index=False)
print(f"\ndf_filtered records: {len(df_filtered):,}")

# ── 5. Profile df_filtered ────────────────────────────────────────────────────
print("\n── df_filtered profile ──────────────────────────────────────────")
print(f"Shape: {df_filtered.shape}")
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
print(f"Date range: {df_filtered['date'].min()} → {df_filtered['date'].max()}")
print(f"\nNull counts:")
print(df_filtered.isnull().sum().to_string())
print(f"\nNumeric summary:")
print(df_filtered.describe().to_string())
print(f"\ndtypes:")
print(df_filtered.dtypes.to_string())
