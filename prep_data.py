#!/usr/bin/env python3
# prep_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-8
#
# Prepare Chicago crime data for training ML models to forecast crimes across that city.

import pandas as pd
import numpy as np

# clear any cached .pyc files
import importlib
import sys

np.random.seed(42)

###
### Load data
###

print('\n=== Loading data/crimes.csv into df_filtered ===')
df_filtered = pd.read_csv('data/crimes.csv', low_memory=False)
print(f'df_filtered shape: {df_filtered.shape}')

###
### Profile df_filtered
###

print('\n=== Profile of df_filtered ===')
print(df_filtered.dtypes)
print()
print(df_filtered.describe(include='all'))
print()
print(f'Missing values per column:\n{df_filtered.isnull().sum()}')

###
### Rename 'date' to 'timestamp', derive 'month' column
###

print('\n=== Renaming date -> timestamp, deriving month ===')
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['month'] = df_filtered['timestamp'].dt.month

###
### Keep only top 20 primary_types
###

print('\n=== Filtering to top 20 primary_types ===')
top20 = df_filtered['primary_type'].value_counts().nlargest(20).index
df_20 = df_filtered[df_filtered['primary_type'].isin(top20)].copy()
print(f'Number of records in df_20: {len(df_20)}')

###
### Display counts of primary_type in df_20
###

print('\n=== Counts of primary_type in df_20 ===')
print(df_20['primary_type'].value_counts())

###
### Display 1 random record in df_20
###

print('\n=== 1 random record in df_20 (all columns) ===')
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(df_20.sample(1).to_string())

###
### Column types in df_20
###

print('\n=== Column types in df_20 ===')
print(df_20.dtypes)

###
### Min and max date
###

print('\n=== Min and max date ===')
print(f'min date: {df_20["timestamp"].min()}')
print(f'max date: {df_20["timestamp"].max()}')

###
### Group by year, month, ward, primary_type and aggregate
###

print('\n=== Grouping into df_avg ===')
df_avg = df_20.groupby(['year', 'month', 'ward', 'primary_type']).agg(
    date=('timestamp', 'min'),
    arrest=('arrest', lambda x: (x == True).mean() if x.dtype == object else x.astype(float).mean()),
    domestic=('domestic', lambda x: (x == True).mean() if x.dtype == object else x.astype(float).mean()),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
    count_0=('id', 'count'),
).reset_index()

# Cast ward as integer
df_avg['ward'] = df_avg['ward'].astype(int)

# Reorder columns: date first, count_0 last
cols = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude', 'count_0']
df_avg = df_avg[cols]

# Order by date, ward, primary_type
df_avg = df_avg.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'Number of records in df_avg: {len(df_avg)}')

###
### Which ward has greatest sum(count_0) for THEFT?
###

print('\n=== Ward with greatest sum(count_0) for THEFT ===')
theft_by_ward = df_avg[df_avg['primary_type'] == 'THEFT'].groupby('ward')['count_0'].sum()
best_ward = theft_by_ward.idxmax()
print(f'Ward with most THEFT: {best_ward} (count_0 sum = {theft_by_ward[best_ward]})')

###
### Show THEFT ward=42 records
###

print('\n=== Records in df_avg with primary_type=THEFT and ward=42 ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_avg[(df_avg['primary_type'] == 'THEFT') & (df_avg['ward'] == 42)].to_string())

###
### Show ARSON ward=42 records
###

print('\n=== Records in df_avg with primary_type=ARSON and ward=42 ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_avg[(df_avg['primary_type'] == 'ARSON') & (df_avg['ward'] == 42)].to_string())

###
### Zero-pad missing records in df_avg -> df_pad
###

print('\n=== Zero-padding missing records -> df_pad ===')

years = df_avg['year'].unique()
months = df_avg['month'].unique()
wards = df_avg['ward'].unique()
primary_types = df_avg['primary_type'].unique()

# Build full index of all combinations
full_index = pd.MultiIndex.from_product(
    [sorted(years), sorted(months), sorted(wards), sorted(primary_types)],
    names=['year', 'month', 'ward', 'primary_type']
)
df_full = pd.DataFrame(index=full_index).reset_index()

# Merge with df_avg
df_pad = df_full.merge(df_avg, on=['year', 'month', 'ward', 'primary_type'], how='left')

# Fill missing count_0 with 0
df_pad['count_0'] = df_pad['count_0'].fillna(0).astype(int)

# Fill missing date from year+month
mask = df_pad['date'].isna()
df_pad.loc[mask, 'date'] = pd.to_datetime(
    df_pad.loc[mask, 'year'].astype(str) + '-' +
    df_pad.loc[mask, 'month'].astype(str).str.zfill(2) + '-01'
)

# Reorder columns: date first, count_0 last
cols = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude', 'count_0']
df_pad = df_pad[cols]

# Order by date, ward, primary_type
df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'Number of records in df_pad: {len(df_pad)}')

###
### Show ARSON ward=42 in df_pad
###

print('\n=== Records in df_pad with primary_type=ARSON and ward=42 ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_pad[(df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)].to_string())

###
### Replace NaN values with random selections of non-NaN values -> df_nan
###

print('\n=== Replacing NaN values with random non-NaN selections -> df_nan ===')
df_nan = df_pad.copy()
for col in df_nan.columns:
    n_null = df_nan[col].isna().sum()
    if n_null > 0:
        non_null_vals = df_nan[col].dropna().values
        fill_vals = np.random.choice(non_null_vals, size=n_null, replace=True)
        df_nan.loc[df_nan[col].isna(), col] = fill_vals
        print(f'  Filled {n_null} NaN values in column "{col}"')

###
### Show ARSON ward=22 in df_nan
###

print('\n=== Records in df_nan with primary_type=ARSON and ward=22 ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_nan[(df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)].to_string())

###
### df_null = df_nan (note: instructions reference df_null but mean df_nan)
###

df_null = df_nan.copy()

print('\n=== 5 random records in df_null ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_null.sample(5).to_string())

###
### Partition by year, month, ward, primary_type and compute shifted counts -> df_target
###

print('\n=== Computing shifted count columns -> df_target ===')

df_target = df_null.copy()
df_target = df_target.sort_values(['year', 'month', 'ward', 'primary_type', 'date']).reset_index(drop=True)

# Within each (ward, primary_type) group, sort by date and compute shifts
df_target = df_target.sort_values(['ward', 'primary_type', 'date']).reset_index(drop=True)

grp = df_target.groupby(['ward', 'primary_type'])

df_target['count_previous'] = grp['count_0'].shift(1)   # 1 month back
df_target['count_1'] = grp['count_0'].shift(-1)          # 1 month forward
df_target['count_2'] = grp['count_0'].shift(-2)          # 2 months forward
df_target['count_3'] = grp['count_0'].shift(-3)          # 3 months forward
df_target['count_4'] = grp['count_0'].shift(-4)          # 4 months forward

df_target['delta_count'] = df_target['count_0'] - df_target['count_previous']

# Reorder columns: date first, then delta_count, count_0, count_1, count_2, count_3, count_4 last
# keep count_previous so it can be dropped later at the dt_monthly step
front_cols = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude', 'count_previous']
end_cols = ['delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']
df_target = df_target[front_cols + end_cols]

df_target = df_target.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

###
### Print THEFT ward=27 records in df_target
###

print('\n=== Records in df_target with primary_type=THEFT and ward=27 ===')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df_target[(df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)].to_string())

###
### Add ran_num and TTVFI columns -> df_tt
###

print('\n=== Adding ran_num and TTVFI -> df_tt ===')
df_tt = df_target.copy()
df_tt['ran_num'] = np.random.uniform(0, 1, size=len(df_tt))
df_tt['TTVFI'] = np.where(df_tt['ran_num'] <= 0.667, 'train', 'test')

###
### Set TTVFI='validate' for last 6 dates -> df_ttv
###

print('\n=== Setting TTVFI=validate for last 6 dates -> df_ttv ===')
df_ttv = df_tt.copy()
last_few_dates = df_ttv['date'].nlargest(6).unique()
# get the 6 largest unique dates
unique_dates = sorted(df_ttv['date'].unique())
last_few_dates = unique_dates[-6:]
df_ttv.loc[df_ttv['date'].isin(last_few_dates), 'TTVFI'] = 'validate'

###
### Set TTVFI='forecast' for last 2 dates -> df_ttvf
###

print('\n=== Setting TTVFI=forecast for last 2 dates -> df_ttvf ===')
df_ttvf = df_ttv.copy()
unique_dates2 = sorted(df_ttvf['date'].unique())
last_few_dates2 = unique_dates2[-2:]
df_ttvf.loc[df_ttvf['date'].isin(last_few_dates2), 'TTVFI'] = 'forecast'

###
### Set TTVFI='incomplete' for last 1 date -> df_ttvfi
###

print('\n=== Setting TTVFI=incomplete for last 1 date -> df_ttvfi ===')
df_ttvfi = df_ttvf.copy()
unique_dates3 = sorted(df_ttvfi['date'].unique())
last_date = [unique_dates3[-1]]
df_ttvfi.loc[df_ttvfi['date'].isin(last_date), 'TTVFI'] = 'incomplete'

###
### Drop columns -> dt_monthly
###

print('\n=== Dropping columns -> dt_monthly ===')
drop_cols = ['arrest', 'domestic', 'count_previous', 'ran_num']
dt_monthly = df_ttvfi.drop(columns=drop_cols)

###
### Drop records having delta_count=NaN
###

print('\n=== Dropping records with delta_count=NaN ===')
before = len(dt_monthly)
dt_monthly = dt_monthly.dropna(subset=['delta_count']).reset_index(drop=True)
after = len(dt_monthly)
print(f'Dropped {before - after} records with NaN delta_count, {after} records remain')

###
### Prettyprint THEFT ward=22 records
###

print('\n=== All records in dt_monthly with primary_type=THEFT and ward=22 ===')
subset = dt_monthly[(dt_monthly['primary_type'] == 'THEFT') & (dt_monthly['ward'] == 22)]
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(subset.to_string(index=False))

###
### Save dt_monthly as data/crimes_monthly.csv
###

print('\n=== Saving dt_monthly to data/crimes_monthly.csv ===')
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print(f'Saved {len(dt_monthly)} records to data/crimes_monthly.csv')
print('\nDone.')
