#!/usr/bin/env python3
"""Prepare Chicago crime data for ML forecasting."""

import pandas as pd

# ── 1. Load filtered data ─────────────────────────────────────────────────────
df_filtered = pd.read_csv('data/crimes.csv', low_memory=False)
print(f"df_filtered records: {len(df_filtered):,}")

# ── 2. Rename date→timestamp, derive month, filter top 20 primary_type ────────
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['month'] = df_filtered['timestamp'].dt.month

top20 = df_filtered['primary_type'].value_counts().nlargest(20).index
df_20 = df_filtered[df_filtered['primary_type'].isin(top20)].copy()
print(f"\ndf_20 records: {len(df_20):,}")

# ── 3. Date range ─────────────────────────────────────────────────────────────
date_min = df_20['timestamp'].min()
date_max = df_20['timestamp'].max()
print(f"date_min: {date_min}")
print(f"date_max: {date_max}")

# ── 4. primary_type counts in df_20 ──────────────────────────────────────────
print(f"\nprimary_type counts in df_20:")
print(df_20['primary_type'].value_counts().to_string())

# ── 5. Random record from df_20 ───────────────────────────────────────────────
print(f"\nRandom record from df_20:")
print(df_20.sample(1).T.to_string())

# ── 6. Column types in df_20 ─────────────────────────────────────────────────
print(f"\ndf_20 dtypes:")
print(df_20.dtypes.to_string())

# ── 7. Group by year, month, ward, primary_type → df_avg ─────────────────────
df_avg = (
    df_20.groupby(['year', 'month', 'ward', 'primary_type'], as_index=False)
    .agg(
        date=('timestamp', lambda x: x.dt.to_period('M').min().to_timestamp()),
        arrest=('arrest', 'mean'),
        domestic=('domestic', 'mean'),
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        count_0=('id', 'count'),
    )
)

# Strip any residual prefix/suffix artifacts, cast ward to int
df_avg.columns = [c.replace('mean_', '').replace('_mean', '') for c in df_avg.columns]
df_avg['ward'] = df_avg['ward'].astype(int)

# Reorder: date first, count_0 last
other_cols = [c for c in df_avg.columns if c not in ('date', 'count_0')]
df_avg = df_avg[['date'] + other_cols + ['count_0']]

# Sort
df_avg = df_avg.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f"\ndf_avg records: {len(df_avg):,}")
print(f"\ndf_avg head:")
print(df_avg.head(10).to_string())

# ── 8. Ward with greatest sum(count_0) for THEFT ─────────────────────────────
theft_by_ward = (
    df_avg[df_avg['primary_type'] == 'THEFT']
    .groupby('ward')['count_0'].sum()
    .sort_values(ascending=False)
)
print(f"\nTop wards by sum(count_0) for THEFT:")
print(theft_by_ward.head(10).to_string())
print(f"\nWard with greatest THEFT count_0: {theft_by_ward.idxmax()} ({theft_by_ward.max():,})")

# ── 9. All THEFT records for ward 42 ─────────────────────────────────────────
mask = (df_avg['primary_type'] == 'THEFT') & (df_avg['ward'] == 42)
print(f"\ndf_avg: primary_type=THEFT, ward=42:")
print(df_avg[mask].to_string())

# ── 10. All ARSON records for ward 42 ────────────────────────────────────────
mask = (df_avg['primary_type'] == 'ARSON') & (df_avg['ward'] == 42)
print(f"\ndf_avg: primary_type=ARSON, ward=42:")
print(df_avg[mask].to_string())

# ── 11. Zero-pad missing year/month/ward/primary_type combinations ────────────
year_months = df_avg[['year', 'month']].drop_duplicates()
wards = df_avg[['ward']].drop_duplicates()
primary_types = df_avg[['primary_type']].drop_duplicates()

year_months['_key'] = 1
wards['_key'] = 1
primary_types['_key'] = 1

full_index = (
    year_months
    .merge(wards, on='_key')
    .merge(primary_types, on='_key')
    .drop(columns='_key')
)

df_pad = full_index.merge(df_avg, on=['year', 'month', 'ward', 'primary_type'], how='left')
df_pad['count_0'] = df_pad['count_0'].fillna(0).astype(int)
df_pad['date'] = pd.to_datetime(df_pad[['year', 'month']].assign(day=1))
df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f"\ndf_pad records: {len(df_pad):,}")

# ── 12. ARSON ward 42 in df_pad ───────────────────────────────────────────────
mask = (df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)
print(f"\ndf_pad: primary_type=ARSON, ward=42:")
print(df_pad[mask].to_string())

# ── 13. Replace NaNs with random non-NaN values per column ───────────────────
import numpy as np
rng = np.random.default_rng(42)

df_nan = df_pad.copy()
for col in df_nan.columns:
    null_mask = df_nan[col].isna()
    if null_mask.any():
        non_null_vals = df_nan.loc[~null_mask, col].values
        df_nan.loc[null_mask, col] = rng.choice(non_null_vals, size=null_mask.sum())

print(f"\ndf_nan NaN counts per column:")
print(df_nan.isna().sum().to_string())

# ── 14. ARSON ward 22 in df_nan ───────────────────────────────────────────────
mask = (df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)
print(f"\ndf_nan: primary_type=ARSON, ward=22:")
print(df_nan[mask].to_string())

# ── 15. Null-pad df_nan out to date_max + 4 months ───────────────────────────
from pandas.tseries.offsets import MonthBegin

date_max_padded = (date_max + 4 * MonthBegin()).normalize()
extra_periods = pd.date_range(
    start=date_max.to_period('M').to_timestamp() + MonthBegin(),
    end=date_max_padded,
    freq='MS',
)

wards_u = df_nan[['ward']].drop_duplicates()
primary_types_u = df_nan[['primary_type']].drop_duplicates()

extra_rows = pd.DataFrame({'date': extra_periods})
extra_rows['_key'] = 1
wards_u['_key'] = 1
primary_types_u['_key'] = 1

extra_full = (
    extra_rows
    .merge(wards_u, on='_key')
    .merge(primary_types_u, on='_key')
    .drop(columns='_key')
)
extra_full['year'] = extra_full['date'].dt.year
extra_full['month'] = extra_full['date'].dt.month

df_null = (
    pd.concat([df_nan, extra_full], ignore_index=True)
    .sort_values(['date', 'ward', 'primary_type'])
    .reset_index(drop=True)
)

print(f"\ndate_max_padded: {date_max_padded.date()}")
print(f"df_null records: {len(df_null):,}")
print(f"\ndf_null NaN counts per column:")
print(df_null.isna().sum().to_string())

# ── 16. ARSON ward 22 in df_null ─────────────────────────────────────────────
mask = (df_null['primary_type'] == 'ARSON') & (df_null['ward'] == 22)
print(f"\ndf_null: primary_type=ARSON, ward=22:")
print(df_null[mask].to_string())

# ── 17. 5 random records from df_null ────────────────────────────────────────
print(f"\n5 random records from df_null:")
print(df_null.sample(5, random_state=42).to_string())

# ── 18. Build df_target with lag/lead count columns ───────────────────────────
df_null_sorted = df_null.sort_values(['ward', 'primary_type', 'date']).copy()

def add_shifts(grp):
    grp = grp.sort_values('date')
    grp['count_previous'] = grp['count_0'].shift(1)
    grp['count_1']        = grp['count_0'].shift(-1)
    grp['count_2']        = grp['count_0'].shift(-2)
    grp['count_3']        = grp['count_0'].shift(-3)
    grp['count_4']        = grp['count_0'].shift(-4)
    grp['delta_count']    = grp['count_0'] - grp['count_previous']
    return grp

df_target = (
    df_null_sorted
    .groupby(['ward', 'primary_type'], group_keys=False)
    .apply(add_shifts)
    .sort_values(['date', 'ward', 'primary_type'])
    .reset_index(drop=True)
)

print(f"\ndf_target records: {len(df_target):,}")
print(f"\ndf_target columns: {df_target.columns.tolist()}")
print(f"\ndf_target head:")
print(df_target.head(10).to_string())

# ── 19. THEFT ward 27 in df_target ───────────────────────────────────────────
mask = (df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)
print(f"\ndf_target: primary_type=THEFT, ward=27:")
print(df_target[mask].to_string())

# ── 20. Build df_ttv (add ran_num) and df_ttvf (add TTV) ──────────────────────
date_validate = pd.Timestamp('2025-04-15')

df_ttv = df_target.copy()
df_ttv['ran_num'] = rng.uniform(0, 1, size=len(df_ttv))

conditions = [
    (df_ttv['date'] <  date_validate) & (df_ttv['ran_num'] <= 0.667),
    (df_ttv['date'] <  date_validate) & (df_ttv['ran_num'] >  0.667),
    (df_ttv['date'] >  date_validate) & (df_ttv['date'] <  date_max),
    (df_ttv['date'] >= date_max),
]
choices = ['train', 'test', 'validate', 'forecast']

df_ttvf = df_ttv.copy()
df_ttvf['TTVF'] = np.select(conditions, choices, default=pd.NA)

print(f"\ndate_validate: {date_validate.date()}")
print(f"\nTTVF value counts:")
print(df_ttvf['TTVF'].value_counts().to_string())

# ── 21. BURGLARY ward 22 in df_ttvf ──────────────────────────────────────────
mask = (df_ttvf['primary_type'] == 'BURGLARY') & (df_ttvf['ward'] == 22)
print(f"\ndf_ttvf: primary_type=BURGLARY, ward=22:")
print(df_ttvf[mask].to_string())

# ── 22. Save df_ttvf ──────────────────────────────────────────────────────────
df_ttvf.to_csv('data/df_ttvf.csv', index=False)
print(f"\nSaved data/df_ttvf.csv  ({len(df_ttvf):,} records)")

# ── 23. Build dt_monthly (drop columns, reorder, drop NaN delta_count non-forecast) ──
dt_monthly = df_ttvf.drop(columns=['arrest', 'domestic', 'count_previous', 'ran_num'])

tail_cols = ['delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4', 'TTVF']
other_cols = [c for c in dt_monthly.columns if c not in ['date'] + tail_cols]
dt_monthly = dt_monthly[['date'] + other_cols + tail_cols]

# Drop rows where delta_count is NaN and TTVF is not 'forecast'
drop_mask = dt_monthly['delta_count'].isna() & (dt_monthly['TTVF'] != 'forecast')
dt_monthly = dt_monthly[~drop_mask].reset_index(drop=True)

print(f"\ndt_monthly columns: {dt_monthly.columns.tolist()}")
print(f"\ndt_monthly records: {len(dt_monthly):,}")

# ── 24. Save dt_monthly ───────────────────────────────────────────────────────
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print(f"\nSaved data/crimes_monthly.csv  ({len(dt_monthly):,} records)")

# ── 25. BURGLARY ward 22 in dt_monthly ───────────────────────────────────────
mask = (dt_monthly['primary_type'] == 'BURGLARY') & (dt_monthly['ward'] == 22)
print(f"\ndt_monthly: primary_type=BURGLARY, ward=22:")
print(dt_monthly[mask].to_string())
