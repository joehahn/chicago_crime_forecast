#!/usr/bin/env python3
"""Train ML models to forecast crimes across Chicago."""

import pandas as pd
import itertools
import numpy as np

# ── 1. Load filtered crime data ───────────────────────────────────────────────
df_filtered = pd.read_csv('data/crimes_filtered.csv', low_memory=False)
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
print(f"df_filtered records: {len(df_filtered):,}")
print(f"columns: {df_filtered.columns.tolist()}")
print(f"date range: {df_filtered['date'].min()} → {df_filtered['date'].max()}")

# ── 2. Rename 'date' → 'timestamp' ───────────────────────────────────────────
df_timestamp = df_filtered.rename(columns={'date': 'timestamp'})
print(f"\ndf_timestamp — 3 random records (all columns):")
print(df_timestamp.sample(3).T.to_string())

# ── 3. Derive month; keep top-20 primary_types ───────────────────────────────
df_timestamp['month'] = df_timestamp['timestamp'].dt.month

top20 = df_timestamp['primary_type'].value_counts().nlargest(20).index
df_20 = df_timestamp[df_timestamp['primary_type'].isin(top20)].copy()
print(f"\nTop 20 primary_types: {top20.tolist()}")
print(f"df_20 records: {len(df_20):,}")

# ── 4. Counts of primary_type in df_20 ───────────────────────────────────────
print(f"\nprimary_type counts in df_20:")
print(df_20['primary_type'].value_counts().to_string())

# ── 5. Display 1 random record ────────────────────────────────────────────────
print(f"\n1 random record in df_20 (all columns):")
print(df_20.sample(1).T.to_string())

# ── 6. Group and aggregate into df_avg ───────────────────────────────────────
df_avg = df_20.groupby(['year', 'month', 'ward', 'primary_type']).agg(
    date=('timestamp', lambda x: x.dt.normalize().min()),
    arrest=('arrest', 'mean'),
    domestic=('domestic', 'mean'),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
    count_0=('id', 'count'),
).reset_index()

df_avg['ward'] = df_avg['ward'].astype(int)

cols = ['date', 'year', 'month', 'ward', 'primary_type',
        'arrest', 'domestic', 'latitude', 'longitude', 'count_0']
df_avg = df_avg[cols].sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f"\ndf_avg records: {len(df_avg):,}")
print(f"columns: {df_avg.columns.tolist()}")

# ── 8. Ward with smallest sum(count_0) for BURGLARY ──────────────────────────
burglary = df_avg[df_avg['primary_type'] == 'BURGLARY']
ward_counts = burglary.groupby('ward')['count_0'].sum().sort_values()
print(f"\nWard with smallest sum(count_0) for BURGLARY:")
print(ward_counts.head(5).to_string())

# ── 10. Ward with greatest sum(count_0) for THEFT ────────────────────────────
theft = df_avg[df_avg['primary_type'] == 'THEFT']
theft_ward_counts = theft.groupby('ward')['count_0'].sum().sort_values(ascending=False)
print(f"\nWard with greatest sum(count_0) for THEFT:")
print(theft_ward_counts.head(5).to_string())

# ── 11. THEFT, ward=42, year=2025 ────────────────────────────────────────────
mask = (df_avg['primary_type'] == 'THEFT') & (df_avg['ward'] == 42) & (df_avg['year'] == 2025)
print(f"\nTHEFT, ward=42, year=2025:")
print(df_avg[mask].to_string())

# ── 9. BURGLARY, ward=22, year=2025 ──────────────────────────────────────────
mask = (df_avg['primary_type'] == 'BURGLARY') & (df_avg['ward'] == 22) & (df_avg['year'] == 2025)
print(f"\nBURGLARY, ward=22, year=2025:")
print(df_avg[mask].to_string())

# ── 12. Zero-pad missing records into df_pad ──────────────────────────────────
years  = df_avg['year'].unique()
months = range(1, 13)
wards  = df_avg['ward'].unique()
ptypes = df_avg['primary_type'].unique()

full_index = pd.DataFrame(
    list(itertools.product(years, months, wards, ptypes)),
    columns=['year', 'month', 'ward', 'primary_type']
)

df_pad = full_index.merge(
    df_avg, on=['year', 'month', 'ward', 'primary_type'], how='left'
)
df_pad['count_0'] = df_pad['count_0'].fillna(0)

# Fill missing date/year/month for zero-padded rows using the first day of each year/month
missing = df_pad['date'].isna()
df_pad.loc[missing, 'date'] = pd.to_datetime(
    df_pad.loc[missing, 'year'].astype(str) + '-' +
    df_pad.loc[missing, 'month'].astype(str).str.zfill(2) + '-01'
)
# year and month are merge keys so already filled; make explicit for clarity
df_pad['year']  = df_pad['year'].astype(int)
df_pad['month'] = df_pad['month'].astype(int)

df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)
print(f"\ndf_pad records: {len(df_pad):,}")

# ── 13. BURGLARY, ward=22, year=2025 in df_pad ───────────────────────────────
mask = (df_pad['primary_type'] == 'BURGLARY') & (df_pad['ward'] == 22) & (df_pad['year'] == 2025)
print(f"\ndf_pad — BURGLARY, ward=22, year=2025:")
print(df_pad[mask].to_string())

# ── 14. Replace NaN with random non-NaN values → df_nan ─────────────────────
df_nan = df_pad.copy()
for col in df_nan.columns:
    if df_nan[col].isna().any():
        non_nan = df_nan[col].dropna().values
        n_missing = df_nan[col].isna().sum()
        df_nan.loc[df_nan[col].isna(), col] = np.random.choice(non_nan, size=n_missing)

mask = (df_nan['primary_type'] == 'BURGLARY') & (df_nan['ward'] == 22) & (df_nan['year'] == 2025)
print(f"\ndf_nan — BURGLARY, ward=22, year=2025:")
print(df_nan[mask].to_string())

# ── 15. 5 random records from df_nan ─────────────────────────────────────────
print(f"\n5 random records from df_nan:")
print(df_nan.sample(5).to_string())

# ── 16. Append count_1, count_2, count_3 (future month shifts) → df_target ───
df_target = (
    df_nan
    .sort_values(['ward', 'primary_type', 'year', 'month'])
    .copy()
)
df_target['count_1'] = df_target.groupby(['ward', 'primary_type'])['count_0'].shift(-1)
df_target['count_2'] = df_target.groupby(['ward', 'primary_type'])['count_0'].shift(-2)
df_target['count_3'] = df_target.groupby(['ward', 'primary_type'])['count_0'].shift(-3)
df_target['count_4'] = df_target.groupby(['ward', 'primary_type'])['count_0'].shift(-4)

# ── 17. THEFT, ward=27, year>=2025 ───────────────────────────────────────────
mask = (df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27) & (df_target['year'] >= 2025)
print(f"\ndf_target — THEFT, ward=27, year>=2025:")
print(df_target[mask].to_string())

# ── 18. TTV split ─────────────────────────────────────────────────────────────
np.random.seed(42)
df_ttv = df_target.copy()
df_ttv['ran_num'] = np.random.uniform(0, 1, size=len(df_ttv))

cutoff = pd.Timestamp('2025-04-01')
df_ttv['TTV'] = 'validate'
before = df_ttv['date'].notna() & (df_ttv['date'] <= cutoff)
df_ttv.loc[before & (df_ttv['ran_num'] <= 0.667), 'TTV'] = 'train'
df_ttv.loc[before & (df_ttv['ran_num'] >  0.667), 'TTV'] = 'test'

# ── 20. BURGLARY, ward=22, year>=2024 in df_ttv ──────────────────────────────
mask = (df_ttv['primary_type'] == 'BURGLARY') & (df_ttv['ward'] == 22) & (df_ttv['year'] >= 2024)
print(f"\ndf_ttv — BURGLARY, ward=22, year>=2024:")
print(df_ttv[mask].to_string())

# ── 21. Train / test / validate splits ───────────────────────────────────────
keep_cols = ['year', 'month', 'ward', 'primary_type', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']
df_train    = df_ttv[df_ttv['TTV'] == 'train'][keep_cols].reset_index(drop=True)
df_test     = df_ttv[df_ttv['TTV'] == 'test'][keep_cols].reset_index(drop=True)
df_validate = df_ttv[df_ttv['TTV'] == 'validate'][keep_cols].reset_index(drop=True)
print(f"\ndf_train: {len(df_train):,}  df_test: {len(df_test):,}  df_validate: {len(df_validate):,}")

# ── 22. 5 random records from df_train ───────────────────────────────────────
print(f"\n5 random records from df_train:")
print(df_train.sample(5).to_string())

# ── 23. Train XGBoost models ──────────────────────────────────────────────────
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all_ptypes = pd.concat([df_train, df_test, df_validate])['primary_type'].unique()
le.fit(all_ptypes)

def encode(df):
    d = df.copy()
    d['primary_type'] = le.transform(d['primary_type'])
    return d

feature_cols = ['year', 'month', 'ward', 'primary_type', 'count_0']

X_train = encode(df_train)[feature_cols]
X_test  = encode(df_test)[feature_cols]

for target, name in [('count_1', 'model_1'), ('count_2', 'model_2'), ('count_3', 'model_3'), ('count_4', 'model_4')]:
    mask_tr = df_train[target].notna()
    mask_te = df_test[target].notna()
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric='rmse', early_stopping_rounds=20,
                         random_state=42)
    model.fit(
        X_train[mask_tr], df_train.loc[mask_tr, target],
        eval_set=[(X_test[mask_te], df_test.loc[mask_te, target])],
        verbose=False,
    )
    globals()[name] = model
    print(f"\n{name} best iteration: {model.best_iteration}  best score: {model.best_score:.4f}")

# ── 24. Predictions on df_validate → df_predict ──────────────────────────────
df_predict = df_validate.copy()
X_val = encode(df_validate)[feature_cols]
df_predict['count_1_pred'] = model_1.predict(X_val)
df_predict['count_2_pred'] = model_2.predict(X_val)
df_predict['count_3_pred'] = model_3.predict(X_val)
df_predict['count_4_pred'] = model_4.predict(X_val)

# ── 25. THEFT, ward=27 in df_predict ─────────────────────────────────────────
mask = (df_predict['primary_type'] == 'THEFT') & (df_predict['ward'] == 27)
print(f"\ndf_predict — THEFT, ward=27:")
print(df_predict[mask].to_string())

# ── 26. Dashboard ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=13, cols=1,
    subplot_titles=[
        'Plot 1: model_1 predictions vs actuals (count_1)',
        'Plot 2: model_2 predictions vs actuals (count_2)',
        'Plot 3: model_3 predictions vs actuals (count_3)',
        'Plot 4: model_4 predictions vs actuals (count_4)',
        'Plot 5: model_1 — THEFT / NARCOTICS / ARSON timeseries',
        'Plot 6: model_2 — THEFT / NARCOTICS / ARSON timeseries',
        'Plot 7: model_3 — THEFT / NARCOTICS / ARSON timeseries',
        'Plot 8: model_4 — THEFT / NARCOTICS / ARSON timeseries',
        'Plot 9: model_1 — wards 27/32/38 timeseries',
        'Plot 10: model_2 — wards 27/32/38 timeseries',
        'Plot 11: model_3 — wards 27/32/38 timeseries',
        'Plot 12: model_4 — wards 27/32/38 timeseries',
        'Plot 13: THEFT ward 27 — actuals + 1/2/3/4-month forecasts',
    ],
    vertical_spacing=0.03,
)

# Pull dates for df_predict from df_ttv
date_map = df_ttv[df_ttv['TTV'] == 'validate'][
    ['year', 'month', 'ward', 'primary_type', 'date']
].reset_index(drop=True)
df_predict = df_predict.merge(date_map, on=['year', 'month', 'ward', 'primary_type'], how='left')

log_range = [np.log10(0.8), np.log10(500)]
scatter_colors = 'rgba(31,119,180,0.3)'

for row, (pred_col, actual_col) in enumerate(
    [('count_1_pred', 'count_1'), ('count_2_pred', 'count_2'), ('count_3_pred', 'count_3'), ('count_4_pred', 'count_4')], start=1
):
    mask = (df_predict[pred_col] > 0) & (df_predict[actual_col] > 0)
    sub = df_predict[mask]
    fig.add_trace(go.Scatter(
        x=sub[actual_col], y=sub[pred_col],
        mode='markers', marker=dict(color=scatter_colors, size=4),
        name=f'{pred_col} vs {actual_col}', showlegend=False,
    ), row=row, col=1)
    axis_min, axis_max = 0.8, 500
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min, axis_max],
        mode='lines', line=dict(dash='dash', color='red'), showlegend=False,
    ), row=row, col=1)
    fig.update_xaxes(type='log', range=log_range, row=row, col=1, title_text='actual')
    fig.update_yaxes(type='log', range=log_range, row=row, col=1, title_text='predicted')

# Timeseries helpers
crime_colors = {'THEFT': 'blue', 'NARCOTICS': 'green', 'ARSON': 'red'}
ward_colors  = {27: 'blue', 32: 'green', 38: 'red'}

for row, pred_col in enumerate(['count_1_pred', 'count_2_pred', 'count_3_pred', 'count_4_pred'], start=5):
    actual_col = pred_col.replace('_pred', '')
    for crime, color in crime_colors.items():
        sub = df_predict[df_predict['primary_type'] == crime].groupby('date').agg(
            actual=(actual_col, 'sum'), pred=(pred_col, 'sum')
        ).reset_index()
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['actual'], mode='lines',
            line=dict(color=color), name=crime, showlegend=(row == 5),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pred'], mode='lines',
            line=dict(color=color, dash='dash'), name=f'{crime} pred', showlegend=(row == 5),
        ), row=row, col=1)
    fig.update_yaxes(type='log', row=row, col=1)

for row, pred_col in enumerate(['count_1_pred', 'count_2_pred', 'count_3_pred', 'count_4_pred'], start=9):
    actual_col = pred_col.replace('_pred', '')
    for ward, color in ward_colors.items():
        sub = df_predict[df_predict['ward'] == ward].groupby('date').agg(
            actual=(actual_col, 'sum'), pred=(pred_col, 'sum')
        ).reset_index()
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['actual'], mode='lines',
            line=dict(color=color), name=f'ward {ward}', showlegend=(row == 9),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['pred'], mode='lines',
            line=dict(color=color, dash='dash'), name=f'ward {ward} pred', showlegend=(row == 9),
        ), row=row, col=1)
    fig.update_yaxes(type='log', row=row, col=1)

# Plot 13: THEFT ward 27 actual + shifted forecasts
sub10 = df_predict[(df_predict['primary_type'] == 'THEFT') & (df_predict['ward'] == 27)].copy()
sub10 = sub10.sort_values('date')
fig.add_trace(go.Scatter(x=sub10['date'], y=sub10['count_0'], mode='lines',
    line=dict(color='black'), name='actual', showlegend=True), row=13, col=1)
fig.add_trace(go.Scatter(x=sub10['date'] + pd.DateOffset(months=1), y=sub10['count_1_pred'], mode='lines',
    line=dict(color='blue', dash='dash'), name='+1mo pred', showlegend=True), row=13, col=1)
fig.add_trace(go.Scatter(x=sub10['date'] + pd.DateOffset(months=2), y=sub10['count_2_pred'], mode='lines',
    line=dict(color='green', dash='dash'), name='+2mo pred', showlegend=True), row=13, col=1)
fig.add_trace(go.Scatter(x=sub10['date'] + pd.DateOffset(months=3), y=sub10['count_3_pred'], mode='lines',
    line=dict(color='red', dash='dash'), name='+3mo pred', showlegend=True), row=13, col=1)
fig.add_trace(go.Scatter(x=sub10['date'] + pd.DateOffset(months=4), y=sub10['count_4_pred'], mode='lines',
    line=dict(color='purple', dash='dash'), name='+4mo pred', showlegend=True), row=13, col=1)

fig.update_layout(height=5500, title_text='Chicago Crime Forecast — Model Dashboard')
fig.write_html('model_dashboard.html')
print("\nSaved model_dashboard.html")

# ── 27. THEFT, ward=27 detail ─────────────────────────────────────────────────
sub27 = df_predict[(df_predict['primary_type'] == 'THEFT') & (df_predict['ward'] == 27)].copy()
sub27 = sub27.sort_values('date')
sub27['date+1m'] = sub27['date'] + pd.DateOffset(months=1)
sub27['date+2m'] = sub27['date'] + pd.DateOffset(months=2)
sub27['date+3m'] = sub27['date'] + pd.DateOffset(months=3)
sub27['date+4m'] = sub27['date'] + pd.DateOffset(months=4)
display_cols = ['date', 'count_0', 'date+1m', 'count_1', 'date+2m', 'count_2', 'date+3m', 'count_3', 'date+4m', 'count_4']
print(f"\nTHEFT ward=27 — actuals and targets:")
print(sub27[display_cols].to_string())
