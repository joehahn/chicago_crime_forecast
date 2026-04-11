#!/usr/bin/env python3
# seasonal_model.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-11

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ---- Load data ----
print('\n=== Loading data/crimes_monthly.csv ===')
df_monthly = pd.read_csv('data/crimes_monthly.csv', parse_dates=['date'])

# Derive week-of-year from date (since 'week' column is absent from CSV)
df_monthly['week'] = df_monthly['date'].dt.isocalendar().week.astype(int)

keep_cols = ['year', 'month', 'week', 'ward', 'primary_type',
             'delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']

# ---- Split into TTVFI partitions ----
df_train    = df_monthly[df_monthly['TTVFI'] == 'train'   ][keep_cols].copy()
df_test     = df_monthly[df_monthly['TTVFI'] == 'test'    ][keep_cols].copy()
df_validate = df_monthly[df_monthly['TTVFI'] == 'validate'][keep_cols].copy()
df_forecast = df_monthly[df_monthly['TTVFI'] == 'forecast'][keep_cols].copy()

print(f'\ndf_train    records: {len(df_train)}')
print(f'df_test     records: {len(df_test)}')
print(f'df_validate records: {len(df_validate)}')
print(f'df_forecast records: {len(df_forecast)}')

# ---- 5 random records from df_train ----
print('\n=== 5 random records from df_train ===')
print(df_train.sample(5, random_state=42).to_string(index=False))

# ---- Encode primary_type as integer for XGBoost ----
le = LabelEncoder()
le.fit(df_monthly['primary_type'])

def encode(df):
    d = df.copy()
    d['primary_type'] = le.transform(d['primary_type'])
    return d

features = ['year', 'month', 'week', 'ward', 'primary_type', 'count_0']

X_train = encode(df_train)[features]
X_test  = encode(df_test )[features]

# ---- Train XGBoost models ----
print('\n=== Training XGBoost models ===')

xgb_params = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model_1 = xgb.XGBRegressor(**xgb_params)
model_1.fit(X_train, df_train['count_1'], eval_set=[(X_test, df_test['count_1'])], verbose=False)
print('model_1 trained (predicts count_1)')

model_2 = xgb.XGBRegressor(**xgb_params)
model_2.fit(X_train, df_train['count_2'], eval_set=[(X_test, df_test['count_2'])], verbose=False)
print('model_2 trained (predicts count_2)')

model_3 = xgb.XGBRegressor(**xgb_params)
model_3.fit(X_train, df_train['count_3'], eval_set=[(X_test, df_test['count_3'])], verbose=False)
print('model_3 trained (predicts count_3)')

# ---- Generate predictions on df_validate ----
print('\n=== Generating predictions on df_validate ===')
df_predict = df_validate.copy()

# Also carry date from df_monthly for plotting
df_predict['date'] = df_monthly.loc[
    df_monthly['TTVFI'] == 'validate', 'date'].values

X_val = encode(df_predict)[features]
df_predict['count_1_pred'] = model_1.predict(X_val)
df_predict['count_2_pred'] = model_2.predict(X_val)
df_predict['count_3_pred'] = model_3.predict(X_val)

# ---- Show THEFT ward=27 in df_predict ----
print('\n=== df_predict: primary_type=THEFT, ward=27 ===')
mask_27 = (df_predict['primary_type'] == 'THEFT') & (df_predict['ward'] == 27)
print(df_predict[mask_27].to_string(index=False))

# ============================================================
# ---- Dashboard ----
# ============================================================

def _log_range(lo=0.8, hi=120):
    return [np.log10(lo), np.log10(hi)]

log_lo, log_hi = 0.8, 120

# Helper: filter positive predictions & actuals
def pos(df, actual_col, pred_col):
    m = (df[actual_col] > 0) & (df[pred_col] > 0)
    return df[m]

# Build subplots: 10 rows
fig = make_subplots(
    rows=10, cols=1,
    subplot_titles=[
        'Plot 1: model_1 predictions vs actuals (count_1)',
        'Plot 2: model_2 predictions vs actuals (count_2)',
        'Plot 3: model_3 predictions vs actuals (count_3)',
        'Plot 4: model_1 – THEFT/NARCOTICS/ARSON time series',
        'Plot 5: model_2 – THEFT/NARCOTICS/ARSON time series',
        'Plot 6: model_3 – THEFT/NARCOTICS/ARSON time series',
        'Plot 7: model_1 – ward 27/32/38 time series',
        'Plot 8: model_2 – ward 27/32/38 time series',
        'Plot 9: model_3 – ward 27/32/38 time series',
        'Plot 10: THEFT ward 27 – count_0 and shifted predictions',
    ],
    vertical_spacing=0.05,
)
fig.update_layout(height=5000, showlegend=True,
                  title_text='Seasonal Model Dashboard', title_x=0.5)

# ---- Plots 1–3: scatter predictions vs actuals ----
scatter_specs = [
    (1, 'count_1', 'count_1_pred', 'model_1'),
    (2, 'count_2', 'count_2_pred', 'model_2'),
    (3, 'count_3', 'count_3_pred', 'model_3'),
]
ref_line = np.array([log_lo, log_hi])

for row, act_col, pred_col, mname in scatter_specs:
    df_pos = pos(df_predict, act_col, pred_col)
    fig.add_trace(go.Scatter(
        x=df_pos[act_col], y=df_pos[pred_col],
        mode='markers', marker=dict(size=3, opacity=0.4),
        name=f'{mname} scatter', showlegend=True,
        legendgroup=f'scatter_{row}',
    ), row=row, col=1)
    # y=x dashed
    fig.add_trace(go.Scatter(
        x=ref_line, y=ref_line,
        mode='lines', line=dict(dash='dash', color='red'),
        name='y=x', showlegend=(row == 1),
        legendgroup='yx',
    ), row=row, col=1)
    fig.update_xaxes(type='log', range=[np.log10(log_lo), np.log10(log_hi)],
                     title_text='Actual', row=row, col=1)
    fig.update_yaxes(type='log', range=[np.log10(log_lo), np.log10(log_hi)],
                     title_text='Predicted', row=row, col=1)

# ---- Plots 4–6: crime-type time series ----
crime_colors = {'THEFT': 'blue', 'NARCOTICS': 'orange', 'ARSON': 'green'}
ts_specs = [
    (4, 'count_1', 'count_1_pred', 'model_1'),
    (5, 'count_2', 'count_2_pred', 'model_2'),
    (6, 'count_3', 'count_3_pred', 'model_3'),
]

for row, act_col, pred_col, mname in ts_specs:
    for crime, color in crime_colors.items():
        dfg = df_predict[df_predict['primary_type'] == crime].groupby('date')[
            [act_col, pred_col]].sum().reset_index()
        show = (row == 4)
        fig.add_trace(go.Scatter(
            x=dfg['date'], y=dfg[act_col],
            mode='lines', line=dict(color=color),
            name=f'{crime} actual', showlegend=show,
            legendgroup=f'crime_{crime}',
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=dfg['date'], y=dfg[pred_col],
            mode='lines', line=dict(color=color, dash='dash'),
            name=f'{crime} pred', showlegend=show,
            legendgroup=f'crime_{crime}_pred',
        ), row=row, col=1)
    fig.update_yaxes(type='log', title_text='Count', row=row, col=1)
    fig.update_xaxes(title_text='Date', row=row, col=1)

# ---- Plots 7–9: ward time series ----
ward_colors = {27: 'red', 32: 'purple', 38: 'brown'}
ward_specs = [
    (7, 'count_1', 'count_1_pred', 'model_1'),
    (8, 'count_2', 'count_2_pred', 'model_2'),
    (9, 'count_3', 'count_3_pred', 'model_3'),
]

for row, act_col, pred_col, mname in ward_specs:
    for ward, color in ward_colors.items():
        dfg = df_predict[df_predict['ward'] == ward].groupby('date')[
            [act_col, pred_col]].sum().reset_index()
        show = (row == 7)
        fig.add_trace(go.Scatter(
            x=dfg['date'], y=dfg[act_col],
            mode='lines', line=dict(color=color),
            name=f'ward {ward} actual', showlegend=show,
            legendgroup=f'ward_{ward}',
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=dfg['date'], y=dfg[pred_col],
            mode='lines', line=dict(color=color, dash='dash'),
            name=f'ward {ward} pred', showlegend=show,
            legendgroup=f'ward_{ward}_pred',
        ), row=row, col=1)
    fig.update_yaxes(type='log', title_text='Count', row=row, col=1)
    fig.update_xaxes(title_text='Date', row=row, col=1)

# ---- Plot 10: THEFT ward=27 shifted predictions ----
df_27 = df_predict[mask_27].sort_values('date').copy()

fig.add_trace(go.Scatter(
    x=df_27['date'], y=df_27['count_0'],
    mode='lines+markers', name='count_0 (current)',
    legendgroup='plot10_c0',
), row=10, col=1)

one_week  = pd.Timedelta(weeks=1)
two_weeks = pd.Timedelta(weeks=2)
three_weeks = pd.Timedelta(weeks=3)

fig.add_trace(go.Scatter(
    x=df_27['date'] + one_week, y=df_27['count_1_pred'],
    mode='lines+markers', line=dict(dash='dash'),
    name='count_1_pred (date+1w)', legendgroup='plot10_c1',
), row=10, col=1)
fig.add_trace(go.Scatter(
    x=df_27['date'] + two_weeks, y=df_27['count_2_pred'],
    mode='lines+markers', line=dict(dash='dot'),
    name='count_2_pred (date+2w)', legendgroup='plot10_c2',
), row=10, col=1)
fig.add_trace(go.Scatter(
    x=df_27['date'] + three_weeks, y=df_27['count_3_pred'],
    mode='lines+markers', line=dict(dash='dashdot'),
    name='count_3_pred (date+3w)', legendgroup='plot10_c3',
), row=10, col=1)
fig.update_xaxes(title_text='Date', row=10, col=1)
fig.update_yaxes(title_text='Count', row=10, col=1)

# ---- Save dashboard ----
fig.write_html('model_dashboard.html')
print('\nSaved model_dashboard.html')

# ---- Step 27: THEFT ward=27 with shifted dates ----
print('\n=== THEFT ward=27: date, count_0, shifted predictions ===')
out = df_27[['date', 'count_0',
             'count_1', 'count_1_pred',
             'count_2', 'count_2_pred',
             'count_3', 'count_3_pred']].copy()
out['date+1w'] = out['date'] + one_week
out['date+2w'] = out['date'] + two_weeks
out['date+3w'] = out['date'] + three_weeks

display_cols = ['date', 'count_0',
                'date+1w', 'count_1', 'count_1_pred',
                'date+2w', 'count_2', 'count_2_pred',
                'date+3w', 'count_3', 'count_3_pred']
print(out[display_cols].to_string(index=False))
