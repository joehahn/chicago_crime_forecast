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

train_cols = ['year', 'month', 'ward', 'primary_type',
              'delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']

# ---- Split into TTVFI partitions ----
df_train    = df_monthly[df_monthly['TTVFI'] == 'train'   ][train_cols].copy()
df_test     = df_monthly[df_monthly['TTVFI'] == 'test'    ][train_cols].copy()
df_validate = df_monthly[df_monthly['TTVFI'] == 'validate'][train_cols].copy()
df_forecast = df_monthly[df_monthly['TTVFI'] == 'forecast'][train_cols].copy()

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

train_features = ['year', 'month', 'ward', 'primary_type', 'delta_count', 'count_0']

X_train = encode(df_train)[train_features]
X_test  = encode(df_test )[train_features]

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

seasonal_1 = xgb.XGBRegressor(**xgb_params)
seasonal_1.fit(X_train, df_train['count_1'], eval_set=[(X_test, df_test['count_1'])], verbose=False)
print('seasonal_1 trained (predicts count_1)')

seasonal_2 = xgb.XGBRegressor(**xgb_params)
seasonal_2.fit(X_train, df_train['count_2'], eval_set=[(X_test, df_test['count_2'])], verbose=False)
print('seasonal_2 trained (predicts count_2)')

seasonal_3 = xgb.XGBRegressor(**xgb_params)
seasonal_3.fit(X_train, df_train['count_3'], eval_set=[(X_test, df_test['count_3'])], verbose=False)
print('seasonal_3 trained (predicts count_3)')

seasonal_4 = xgb.XGBRegressor(**xgb_params)
seasonal_4.fit(X_train, df_train['count_4'], eval_set=[(X_test, df_test['count_4'])], verbose=False)
print('seasonal_4 trained (predicts count_4)')

# ---- Feature importance ----
print('\n=== Feature importance ===')
fi_df = pd.DataFrame({'feature': train_features})
for name, model in [('seasonal_1', seasonal_1), ('seasonal_2', seasonal_2),
                    ('seasonal_3', seasonal_3), ('seasonal_4', seasonal_4)]:
    fi_df[name] = model.feature_importances_
print(fi_df.to_string(index=False))

# ---- Generate predictions on df_test ----
print('\n=== Generating predictions on df_test ===')
X_test_enc = encode(df_test)[train_features]
df_test['count_1_pred'] = seasonal_1.predict(X_test_enc)
df_test['count_2_pred'] = seasonal_2.predict(X_test_enc)
df_test['count_3_pred'] = seasonal_3.predict(X_test_enc)
df_test['count_4_pred'] = seasonal_4.predict(X_test_enc)
print(f'df_test predictions appended: {len(df_test)} records')
print(df_test.head(5).to_string(index=False))

# ---- Generate predictions on df_validate ----
print('\n=== Generating predictions on df_validate ===')
X_val_enc = encode(df_validate)[train_features]
df_validate['count_1_pred'] = seasonal_1.predict(X_val_enc)
df_validate['count_2_pred'] = seasonal_2.predict(X_val_enc)
df_validate['count_3_pred'] = seasonal_3.predict(X_val_enc)
df_validate['count_4_pred'] = seasonal_4.predict(X_val_enc)
print(f'df_validate predictions appended: {len(df_validate)} records')
print(df_validate.head(5).to_string(index=False))

df_predict = df_validate.copy()

# Also carry date from df_monthly for plotting
df_predict['date'] = df_monthly.loc[
    df_monthly['TTVFI'] == 'validate', 'date'].values

X_val = encode(df_predict)[train_features]
df_predict['count_1_pred'] = seasonal_1.predict(X_val)
df_predict['count_2_pred'] = seasonal_2.predict(X_val)
df_predict['count_3_pred'] = seasonal_3.predict(X_val)
df_predict['count_4_pred'] = seasonal_4.predict(X_val)

# ---- Show THEFT ward=27 in df_predict ----
print('\n=== df_predict: primary_type=THEFT, ward=27 ===')
mask_27 = (df_predict['primary_type'] == 'THEFT') & (df_predict['ward'] == 27)
print(df_predict[mask_27].to_string(index=False))

# ============================================================
# ---- Dashboard ----
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log_lo, log_hi = 0.5, 700

scatter_specs = [
    (3, 'count_1', 'count_1_pred', 'seasonal_1'),
    (4, 'count_2', 'count_2_pred', 'seasonal_2'),
    (5, 'count_3', 'count_3_pred', 'seasonal_3'),
    (6, 'count_4', 'count_4_pred', 'seasonal_4'),
]

# ---- Validation scores (computed on df_test) ----
print('\n=== Validation scores ===')
score_rows = []
for _, act_col, pred_col, mname in scatter_specs:
    df_s = df_test[[act_col, pred_col]].dropna()
    mae  = mean_absolute_error(df_s[act_col], df_s[pred_col])
    rmse = np.sqrt(mean_squared_error(df_s[act_col], df_s[pred_col]))
    r2   = r2_score(df_s[act_col], df_s[pred_col])
    score_rows.append({'model': mname, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
scores_df = pd.DataFrame(score_rows)
print(scores_df.to_string(index=False))

# ---- Build subplots: 2 tables + 4 scatter + 3 timeseries ----
fig = make_subplots(
    rows=9, cols=1,
    subplot_titles=[
        'Table: Validation scores (df_test)',
        'Table: Feature importance',
        'Plot 1: seasonal_1 predictions vs actuals – df_test (count_1)',
        'Plot 2: seasonal_2 predictions vs actuals – df_test (count_2)',
        'Plot 3: seasonal_3 predictions vs actuals – df_test (count_3)',
        'Plot 4: seasonal_4 predictions vs actuals – df_test (count_4)',
        'Plot 5: THEFT – summed count_0 and predictions shifted by 1–4 months (df_validate)',
        'Plot 6: ASSAULT – summed count_0 and predictions shifted by 1–4 months (df_validate)',
        'Plot 7: ARSON – summed count_0 and predictions shifted by 1–4 months (df_validate)',
    ],
    specs=[[{'type': 'table'}], [{'type': 'table'}],
           [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}],
           [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}]],
    vertical_spacing=0.03,
)
fig.update_layout(height=5400, showlegend=True,
                  title_text='Seasonal Model Dashboard', title_x=0.5)

# ---- Table 1: validation scores (row 1) ----
fig.add_trace(go.Table(
    header=dict(values=['Model', 'MAE', 'RMSE', 'R²'],
                fill_color='steelblue', font=dict(color='white'), align='center'),
    cells=dict(
        values=[scores_df['model'], scores_df['MAE'].round(3),
                scores_df['RMSE'].round(3), scores_df['R2'].round(3)],
        align='center',
    ),
), row=1, col=1)

# ---- Table 2: feature importance (row 2) ----
fi_str = fi_df.copy()
for col in ['seasonal_1', 'seasonal_2', 'seasonal_3', 'seasonal_4']:
    fi_str[col] = fi_str[col].astype(str).str[:5]
fig.add_trace(go.Table(
    header=dict(values=['Feature', 'seasonal_1', 'seasonal_2', 'seasonal_3', 'seasonal_4'],
                fill_color='steelblue', font=dict(color='white'), align='center'),
    cells=dict(
        values=[fi_str['feature'], fi_str['seasonal_1'], fi_str['seasonal_2'],
                fi_str['seasonal_3'], fi_str['seasonal_4']],
        align='center',
    ),
), row=2, col=1)

# ---- Plots 1-4: scatter df_test predictions vs actuals ----
ref_line = np.array([log_lo, log_hi])
for row, act_col, pred_col, mname in scatter_specs:
    df_pos = df_test[(df_test[act_col] > 0) & (df_test[pred_col] > 0)]
    fig.add_trace(go.Scatter(
        x=df_pos[act_col], y=df_pos[pred_col],
        mode='markers', marker=dict(size=3, opacity=1.0, color='blue'),
        name=f'{mname} scatter', showlegend=True,
        legendgroup=f'scatter_{row}',
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=ref_line, y=ref_line,
        mode='lines', line=dict(dash='dash', color='red'),
        name='y=x', showlegend=(row == 3),
        legendgroup='yx',
    ), row=row, col=1)
    fig.update_xaxes(type='log', range=[np.log10(log_lo), np.log10(log_hi)],
                     title_text='Actual', row=row, col=1)
    fig.update_yaxes(type='log', range=[np.log10(log_lo), np.log10(log_hi)],
                     title_text='Predicted', row=row, col=1)

# ---- attach date to df_validate for timeseries ----
df_validate_dated = df_validate.copy()
df_validate_dated['date'] = df_monthly.loc[
    df_monthly['TTVFI'] == 'validate', 'date'].values

# ---- Helper: add shifted-prediction timeseries for one crime type ----
def add_crime_timeseries(crime, row):
    dfc = df_validate_dated[df_validate_dated['primary_type'] == crime].groupby('date')[
        ['count_0', 'count_1_pred', 'count_2_pred', 'count_3_pred', 'count_4_pred']
    ].sum().reset_index()
    show = (row == 7)
    fig.add_trace(go.Scatter(
        x=dfc['date'], y=dfc['count_0'],
        mode='lines+markers', line=dict(color='blue'),
        name='count_0 actual', legendgroup='ts_c0', showlegend=show,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=1), y=dfc['count_1_pred'],
        mode='lines+markers', line=dict(color='orange', dash='dash'),
        name='count_1_pred (date+1m)', legendgroup='ts_c1', showlegend=show,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=2), y=dfc['count_2_pred'],
        mode='lines+markers', line=dict(color='green', dash='dash'),
        name='count_2_pred (date+2m)', legendgroup='ts_c2', showlegend=show,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=3), y=dfc['count_3_pred'],
        mode='lines+markers', line=dict(color='red', dash='dash'),
        name='count_3_pred (date+3m)', legendgroup='ts_c3', showlegend=show,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=4), y=dfc['count_4_pred'],
        mode='lines+markers', line=dict(color='purple', dash='dash'),
        name='count_4_pred (date+4m)', legendgroup='ts_c4', showlegend=show,
    ), row=row, col=1)
    fig.update_yaxes(title_text='Count', row=row, col=1)
    fig.update_xaxes(title_text='Date', row=row, col=1)

# ---- Plots 5-7: crime timeseries ----
add_crime_timeseries('THEFT',   row=7)
add_crime_timeseries('ASSAULT', row=8)
add_crime_timeseries('ARSON',   row=9)

# ---- Legend in lower-right of plot 5 ----
fig.update_layout(legend=dict(x=1.0, y=0.22, xanchor='right', yanchor='bottom',
                               bgcolor='rgba(255,255,255,0.8)', bordercolor='gray',
                               borderwidth=1))

# ---- Save dashboard ----
fig.write_html('seasonal_model_dashboard.html')
print('\nSaved seasonal_model_dashboard.html')
