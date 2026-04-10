#!/usr/bin/env python3
# prep_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-9

import pandas as pd
import numpy as np
from itertools import product

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ---- Load and profile df_filtered ----
df_filtered = pd.read_csv('data/crimes.csv')
print('=== df_filtered profile ===')
print(f'shape: {df_filtered.shape}')
print(df_filtered.dtypes)
print(df_filtered.describe())
print()

# ---- Rename date -> timestamp, derive month ----
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['month'] = df_filtered['timestamp'].dt.month

# ---- Keep only top 20 primary_types ----
top20 = df_filtered['primary_type'].value_counts().nlargest(20).index.tolist()
df_20 = df_filtered[df_filtered['primary_type'].isin(top20)].copy()
print(f'df_20 records: {len(df_20):,}')

# ---- date_min, date_max ----
date_min = df_20['timestamp'].min()
date_max = df_20['timestamp'].max()
print(f'date_min: {date_min}')
print(f'date_max: {date_max}')

# ---- Counts of primary_type ----
print('\nCounts of primary_type in df_20:')
print(df_20['primary_type'].value_counts())

# ---- 1 random record ----
print('\n1 random record in df_20 (all columns):')
print(df_20.sample(1, random_state=42).to_string())

# ---- Column types ----
print('\nColumn types in df_20:')
print(df_20.dtypes)

# ---- Convert arrest/domestic to int for mean ----
df_20['arrest'] = df_20['arrest'].astype(int)
df_20['domestic'] = df_20['domestic'].astype(int)

# ---- Group to df_avg ----
df_avg = df_20.groupby(['year', 'month', 'ward', 'primary_type']).agg(
    arrest=('arrest', 'mean'),
    domestic=('domestic', 'mean'),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
    count_0=('id', 'count'),
).reset_index()

# Construct date as first day of month
df_avg['date'] = pd.to_datetime({'year': df_avg['year'], 'month': df_avg['month'], 'day': 1})

# Cast ward as integer
df_avg['ward'] = df_avg['ward'].astype(int)

# Reorder columns: date first, count_0 last
cols = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude', 'count_0']
df_avg = df_avg[cols]

# Order by date, ward, primary_type
df_avg = df_avg.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'\ndf_avg records: {len(df_avg):,}')

# ---- Which ward has greatest sum(count_0) for THEFT? ----
theft_sums = df_avg[df_avg['primary_type'] == 'THEFT'].groupby('ward')['count_0'].sum()
best_theft_ward = theft_sums.idxmax()
print(f'\nWard with greatest THEFT sum(count_0): ward={best_theft_ward} (total={theft_sums[best_theft_ward]:,})')

# ---- Show THEFT ward=42 ----
print('\ndf_avg: primary_type=THEFT, ward=42:')
print(df_avg[(df_avg['primary_type'] == 'THEFT') & (df_avg['ward'] == 42)].to_string())

# ---- Show ARSON ward=42 ----
print('\ndf_avg: primary_type=ARSON, ward=42:')
print(df_avg[(df_avg['primary_type'] == 'ARSON') & (df_avg['ward'] == 42)].to_string())

# ---- Zero-pad missing (year, month, ward, primary_type) combinations ----
date_min_month = pd.Timestamp(date_min.year, date_min.month, 1)
date_max_month = pd.Timestamp(date_max.year, date_max.month, 1)

all_months = pd.date_range(start=date_min_month, end=date_max_month, freq='MS')
all_wards = sorted(df_avg['ward'].unique())
all_types = sorted(df_avg['primary_type'].unique())

# Full cartesian product of dates x wards x types
full_index = pd.MultiIndex.from_product([all_months, all_wards, all_types],
                                         names=['date', 'ward', 'primary_type'])
df_full = pd.DataFrame(index=full_index).reset_index()
df_full['year'] = df_full['date'].dt.year
df_full['month'] = df_full['date'].dt.month

# Identify missing combinations
df_avg_keys = df_avg[['date', 'ward', 'primary_type']].copy()
df_avg_keys['_exists'] = True
df_full = df_full.merge(df_avg_keys, on=['date', 'ward', 'primary_type'], how='left')

df_missing = df_full[df_full['_exists'].isna()].drop(columns=['_exists']).copy()
df_missing['arrest'] = np.nan
df_missing['domestic'] = np.nan
df_missing['latitude'] = np.nan
df_missing['longitude'] = np.nan
df_missing['count_0'] = 0

# Combine and reorder
df_pad = pd.concat([df_avg, df_missing], ignore_index=True)
df_pad = df_pad[cols]
df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'\ndf_pad records: {len(df_pad):,}')

print('\ndf_pad: primary_type=ARSON, ward=42:')
print(df_pad[(df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)].to_string())

# ---- Fill NaN values with random non-NaN selections ----
df_nan = df_pad.copy()
for col in df_nan.columns:
    if df_nan[col].isna().any():
        non_nan = df_nan[col].dropna().values
        if len(non_nan) > 0:
            nan_mask = df_nan[col].isna()
            df_nan.loc[nan_mask, col] = np.random.choice(non_nan, size=nan_mask.sum())

print('\ndf_nan: primary_type=ARSON, ward=22:')
print(df_nan[(df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)].to_string())

# ---- NaN-pad out to date_max + 4 months ----
future_months = pd.date_range(
    start=date_max_month + pd.DateOffset(months=1),
    periods=4,
    freq='MS'
)

future_rows = [
    {'date': dt, 'year': dt.year, 'month': dt.month,
     'ward': ward, 'primary_type': pt,
     'arrest': np.nan, 'domestic': np.nan,
     'latitude': np.nan, 'longitude': np.nan,
     'count_0': np.nan}
    for dt, ward, pt in product(future_months, all_wards, all_types)
]

df_null = pd.concat([df_nan, pd.DataFrame(future_rows)], ignore_index=True)
df_null = df_null[cols]
df_null = df_null.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print('\ndf_null: primary_type=ARSON, ward=22:')
print(df_null[(df_null['primary_type'] == 'ARSON') & (df_null['ward'] == 22)].to_string())

print('\n5 random records in df_null:')
print(df_null.sample(5, random_state=42).to_string())

# ---- Time-shifted columns (partition by ward, primary_type; order by date) ----
df_target = (
    df_null
    .sort_values(['ward', 'primary_type', 'date'])
    .groupby(['ward', 'primary_type'], group_keys=False)
    .apply(lambda g: g.assign(
        count_previous=g['count_0'].shift(1),
        count_1=g['count_0'].shift(-1),
        count_2=g['count_0'].shift(-2),
        count_3=g['count_0'].shift(-3),
        count_4=g['count_0'].shift(-4),
    ))
)
df_target['delta_count'] = df_target['count_0'] - df_target['count_previous']
df_target = df_target.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print('\ndf_target: primary_type=THEFT, ward=27:')
print(df_target[(df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)].to_string())

# ---- TTVF labels ----
df_ttv = df_target.copy()
df_ttv['ran_num'] = np.random.uniform(0, 1, len(df_ttv))

date_validate = pd.Timestamp(date_max.year, date_max.month, 1)
date_tt = date_validate - pd.DateOffset(months=5)
print(f'date_min:      {date_min}')
print(f'date_max:      {date_max}')
print(f'date_validate: {date_validate}')
print(f'date_tt:       {date_tt}')

def assign_ttvf(row):
    d = row['date']
    if d >= date_validate:
        return 'forecast'
    elif d > date_tt:
        return 'validate'
    elif row['ran_num'] <= 0.667:
        return 'train'
    else:
        return 'test'

df_ttv['TTVF'] = df_ttv.apply(assign_ttvf, axis=1)

# Drop columns
df_ttvf = df_ttv.drop(columns=['arrest', 'domestic', 'count_previous', 'ran_num'])

# Reorder: date first, target columns and TTVF last
meta_cols = [c for c in df_ttvf.columns
             if c not in ['date', 'delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4', 'TTVF']]
final_cols = ['date'] + meta_cols + ['delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4', 'TTVF']
df_ttvf = df_ttvf[final_cols]

# Drop records where delta_count is NaN AND TTVF != 'forecast'
dt_monthly = df_ttvf[~(df_ttvf['delta_count'].isna() & (df_ttvf['TTVF'] != 'forecast'))].copy()
dt_monthly = dt_monthly.reset_index(drop=True)

print('\ndt_monthly: primary_type=BURGLARY, ward=22:')
print(dt_monthly[(dt_monthly['primary_type'] == 'BURGLARY') & (dt_monthly['ward'] == 22)].to_string())

# ---- Save ----
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print('\nSaved data/crimes_monthly.csv')
