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
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ---- Load data ----
print('\n=== Loading data/crimes_monthly.csv ===')
df_monthly = pd.read_csv('data/crimes_monthly.csv', parse_dates=['date'])

train_cols = ['date', 'year', 'month', 'ward', 'primary_type',
              'delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']

# ---- Split into TTV partitions ----
df_train    = df_monthly[df_monthly['TTV'] == 'train'   ][train_cols].copy()
df_test     = df_monthly[df_monthly['TTV'] == 'test'    ][train_cols].copy()
df_validate = df_monthly[df_monthly['TTV'] == 'validate'][train_cols].copy()
df_forecast = df_monthly[df_monthly['TTV'] == 'forecast'][train_cols].copy()

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

# ---- Save models ----
os.makedirs('models', exist_ok=True)
for name, model in [('seasonal_1', seasonal_1), ('seasonal_2', seasonal_2),
                    ('seasonal_3', seasonal_3), ('seasonal_4', seasonal_4)]:
    path = f'models/{name}.json'
    model.save_model(path)
    print(f'Saved {path}')

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

# ---- Show df_test: primary_type=THEFT, ward=27 ----
print('\n=== df_test: primary_type=THEFT, ward=27 ===')
mask_test_27 = (df_test['primary_type'] == 'THEFT') & (df_test['ward'] == 27)
print(df_test[mask_test_27].to_string(index=False))

# ---- Generate predictions on df_validate ----
print('\n=== Generating predictions on df_validate ===')
X_val_enc = encode(df_validate)[train_features]
df_validate['count_1_pred'] = seasonal_1.predict(X_val_enc)
df_validate['count_2_pred'] = seasonal_2.predict(X_val_enc)
df_validate['count_3_pred'] = seasonal_3.predict(X_val_enc)
df_validate['count_4_pred'] = seasonal_4.predict(X_val_enc)
print(f'df_validate predictions appended: {len(df_validate)} records')
print(df_validate.head(5).to_string(index=False))

# ---- Show df_validate: primary_type=THEFT, ward=27 ----
print('\n=== df_validate: primary_type=THEFT, ward=27 ===')
mask_val_27 = (df_validate['primary_type'] == 'THEFT') & (df_validate['ward'] == 27)
print(df_validate[mask_val_27].to_string(index=False))

# ---- Generate predictions on df_forecast ----
print('\n=== Generating predictions on df_forecast ===')
X_forecast_enc = encode(df_forecast)[train_features]
df_forecast['count_1_pred'] = seasonal_1.predict(X_forecast_enc)
df_forecast['count_2_pred'] = seasonal_2.predict(X_forecast_enc)
df_forecast['count_3_pred'] = seasonal_3.predict(X_forecast_enc)
df_forecast['count_4_pred'] = seasonal_4.predict(X_forecast_enc)
print(f'df_forecast predictions appended: {len(df_forecast)} records')

# ---- Show df_forecast: primary_type=THEFT, ward=27 ----
print('\n=== df_forecast: primary_type=THEFT, ward=27 ===')
mask_fore_27 = (df_forecast['primary_type'] == 'THEFT') & (df_forecast['ward'] == 27)
print(df_forecast[mask_fore_27].to_string(index=False))

# ============================================================
# ---- Dashboard ----
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# scatter axis ranges: x=actual, y=predicted (asymmetric)
x_lo, x_hi = 0.8, 600
y_lo, y_hi = 0.2, 600

# scatter_specs: (subplot_row, actual_col, pred_col, model_name, legend_ref)
scatter_specs = [
    (4, 'count_1', 'count_1_pred', 'seasonal_1', 'legend3'),
    (5, 'count_2', 'count_2_pred', 'seasonal_2', 'legend4'),
    (6, 'count_3', 'count_3_pred', 'seasonal_3', 'legend5'),
    (7, 'count_4', 'count_4_pred', 'seasonal_4', 'legend6'),
]

# ---- Validation scores (df_validate) ----
print('\n=== Validation scores ===')
score_rows = []
for _, act_col, pred_col, mname, _ in scatter_specs:
    df_s = df_validate[[act_col, pred_col]].dropna()
    mae  = mean_absolute_error(df_s[act_col], df_s[pred_col])
    rmse = np.sqrt(mean_squared_error(df_s[act_col], df_s[pred_col]))
    r2   = r2_score(df_s[act_col], df_s[pred_col])
    score_rows.append({'model': mname, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
scores_df = pd.DataFrame(score_rows)
print(scores_df.to_string(index=False))

# ---- Plot 1 data: sum(count_0) by date grouped as train+test / validate / forecast ----
df_all_ts = df_monthly.groupby(['date', 'TTV'])['count_0'].sum().reset_index()
df_all_ts.columns = ['date', 'TTV', 'total_count']
df_trte = (df_all_ts[df_all_ts['TTV'].isin(['train', 'test'])]
           .groupby('date')['total_count'].sum().reset_index())
df_trte['TTV'] = 'train+test'
df_val_ts  = df_all_ts[df_all_ts['TTV'] == 'validate'].copy()
df_fore_ts = df_all_ts[df_all_ts['TTV'] == 'forecast'].copy()

# ---- count_0 percentile thresholds for scatter coloring ----
p10 = df_validate['count_0'].quantile(0.10)
p90 = df_validate['count_0'].quantile(0.90)
print(f'\ncount_0 percentiles in df_validate: p10={p10:.1f}, p90={p90:.1f}')

# ---- Load Plot 10 data: THEFT in March 2026 from crimes.csv ----
print('\n=== Loading Plot 10 data: THEFT March 2026 from crimes.csv ===')
df_crimes = pd.read_csv('data/crimes.csv')
df_crimes['date'] = pd.to_datetime(df_crimes['date'])
df_theft_march = df_crimes[
    (df_crimes['primary_type'] == 'THEFT') &
    (df_crimes['date'].dt.year == 2026) &
    (df_crimes['date'].dt.month == 3)
].dropna(subset=['latitude', 'longitude'])
print(f'THEFT March 2026 records: {len(df_theft_march)}')

# ---- Build subplots: 12 rows ----
# row  1: Plot 1  (TTV overview ts)
# row  2: Table 1 (scores)
# row  3: Table 2 (feature importance)
# rows 4-7: Plots 2-5 (scatter)
# rows 8-10: Plots 6-8 (crime ts with errorbars)
# row 11: Plot 9  (ward ts, log y)
# row 12: Plot 10 (THEFT heatmap on Chicago street map)
fig = make_subplots(
    rows=12, cols=1,
    subplot_titles=[
        'Plot 1: All crimes total_count vs date (train+test / validate / forecast)',
        'Table 1: Validation scores (df_validate)',
        'Table 2: Feature importance',
        'Plot 2: seasonal_1 predictions vs actuals – df_validate (count_1)',
        'Plot 3: seasonal_2 predictions vs actuals – df_validate (count_2)',
        'Plot 4: seasonal_3 predictions vs actuals – df_validate (count_3)',
        'Plot 5: seasonal_4 predictions vs actuals – df_validate (count_4)',
        'Plot 6: THEFT – summed count_0 ± √count_0 and predictions (df_validate)',
        'Plot 7: BURGLARY – summed count_0 ± √count_0 and predictions (df_validate)',
        'Plot 8: ARSON – summed count_0 ± √count_0 and predictions (df_validate)',
        'Plot 9: All crimes – wards 27/29/38 summed count_0 and predictions (df_validate)',
        'Plot 10: THEFT heatmap – Chicago March 2026',
    ],
    specs=[[{'type': 'xy'}],
           [{'type': 'table'}], [{'type': 'table'}],
           [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}],
           [{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'xy'}],
           [{'type': 'xy'}],
           [{'type': 'map'}]],
    vertical_spacing=0.01,
)
fig.update_layout(
    height=9500,
    title_text='Seasonal Model Dashboard', title_x=0.5,
    margin=dict(t=40, b=5, l=40, r=40),
)

# ---- Legend helpers ----
# With 12 rows, vertical_spacing=0.01:
#   total spacing = 11 * 0.01 = 0.11
#   total plot area = 1 - 0.11 = 0.89
#   row_height = 0.89 / 12 ≈ 0.07417
rh = 0.89 / 12
def row_top(r):    return round(1.0 - (r - 1) * (rh + 0.01), 4)
def row_bottom(r): return round(row_top(r) - rh, 4)

def leg_lr(ref, r):   # lower-right
    return {ref: dict(x=0.99, y=row_bottom(r), xanchor='right', yanchor='bottom',
                      bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1)}
def leg_ur(ref, r):   # upper-right
    return {ref: dict(x=0.99, y=row_top(r), xanchor='right', yanchor='top',
                      bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1)}

# ---- Plot 1: train+test / validate / forecast timeseries (row 1) ----
seg_colors = {'train+test': 'steelblue', 'validate': 'green', 'forecast': 'red'}
for df_seg, label in [(df_trte, 'train+test'), (df_val_ts, 'validate'), (df_fore_ts, 'forecast')]:
    seg = df_seg.sort_values('date')
    fig.add_trace(go.Scatter(
        x=seg['date'], y=seg['total_count'],
        mode='lines+markers', marker=dict(size=4),
        line=dict(color=seg_colors[label]),
        name=label, legend='legend', showlegend=True,
    ), row=1, col=1)
fig.update_xaxes(title_text='Date', row=1, col=1)
fig.update_yaxes(title_text='total_count', row=1, col=1)

# ---- Table 1: validation scores (row 2) ----
fig.add_trace(go.Table(
    header=dict(values=['Model', 'MAE', 'RMSE', 'R²'],
                fill_color='steelblue', font=dict(color='white'), align='center'),
    cells=dict(values=[scores_df['model'], scores_df['MAE'].round(3),
                       scores_df['RMSE'].round(3), scores_df['R2'].round(3)],
               align='center'),
), row=2, col=1)

# ---- Table 2: feature importance (row 3) ----
fi_str = fi_df.copy()
for col in ['seasonal_1', 'seasonal_2', 'seasonal_3', 'seasonal_4']:
    fi_str[col] = fi_str[col].astype(str).str[:5]
fig.add_trace(go.Table(
    header=dict(values=['Feature', 'seasonal_1', 'seasonal_2', 'seasonal_3', 'seasonal_4'],
                fill_color='steelblue', font=dict(color='white'), align='center'),
    cells=dict(values=[fi_str['feature'], fi_str['seasonal_1'], fi_str['seasonal_2'],
                       fi_str['seasonal_3'], fi_str['seasonal_4']],
               align='center'),
), row=3, col=1)

# ---- Plots 2-5: scatter df_validate predictions vs actuals (rows 4-7) ----
ref_x = np.array([x_lo, x_hi])
ref_y = np.array([x_lo, x_hi])
for row, act_col, pred_col, mname, leg_ref in scatter_specs:
    df_pos = df_validate[(df_validate[act_col] > 0) & (df_validate[pred_col] > 0)].copy()
    inner = df_pos[(df_pos['count_0'] >= p10) & (df_pos['count_0'] <= p90)]
    outer = df_pos[(df_pos['count_0'] <  p10) | (df_pos['count_0'] >  p90)]
    fig.add_trace(go.Scatter(
        x=inner[act_col], y=inner[pred_col],
        mode='markers', marker=dict(size=3, opacity=1.0, color='green'),
        name='inner 80% (count_0)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=outer[act_col], y=outer[pred_col],
        mode='markers', marker=dict(size=3, opacity=1.0, color='red'),
        name='outer 20% (count_0)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=ref_x, y=ref_y,
        mode='lines', line=dict(dash='dash', color='black'),
        name='prediction=actual', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.update_xaxes(type='log', range=[np.log10(x_lo), np.log10(x_hi)],
                     title_text='Actual', row=row, col=1)
    fig.update_yaxes(type='log', range=[np.log10(y_lo), np.log10(y_hi)],
                     title_text='Predicted', row=row, col=1)

# ---- Plots 6-8: crime timeseries with errorbars (rows 8-10) ----
ts_specs = [
    ('THEFT',    8, 'legend7'),
    ('BURGLARY', 9, 'legend8'),
    ('ARSON',   10, 'legend9'),
]
for crime, row, leg_ref in ts_specs:
    dfc = df_validate[df_validate['primary_type'] == crime].groupby('date')[
        ['count_0', 'count_1_pred', 'count_2_pred', 'count_3_pred', 'count_4_pred']
    ].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=dfc['date'], y=dfc['count_0'],
        mode='lines+markers', line=dict(color='blue'),
        error_y=dict(type='data', array=np.sqrt(dfc['count_0']), visible=True,
                     color='blue', thickness=1, width=4),
        name='count_0 actual', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=1), y=dfc['count_1_pred'],
        mode='lines+markers', line=dict(color='orange', dash='dash'),
        name='count_1_pred (date+1m)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=2), y=dfc['count_2_pred'],
        mode='lines+markers', line=dict(color='green', dash='dash'),
        name='count_2_pred (date+2m)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=3), y=dfc['count_3_pred'],
        mode='lines+markers', line=dict(color='red', dash='dash'),
        name='count_3_pred (date+3m)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=dfc['date'] + pd.DateOffset(months=4), y=dfc['count_4_pred'],
        mode='lines+markers', line=dict(color='purple', dash='dash'),
        name='count_4_pred (date+4m)', legend=leg_ref, showlegend=True,
    ), row=row, col=1)
    fig.update_yaxes(title_text='Count', row=row, col=1)
    fig.update_xaxes(title_text='Date', row=row, col=1)

# ---- Plot 9: all crimes, wards 27/29/38, log y-axis (row 11) ----
ward_colors = {27: 'red', 29: 'blue', 38: 'green'}
pred_cols = ['count_0', 'count_1_pred', 'count_2_pred', 'count_3_pred', 'count_4_pred']
leg_ref9 = 'legend10'
for ward in [27, 29, 38]:
    dfw = (df_validate[df_validate['ward'] == ward]
           .groupby('date')[pred_cols].sum().reset_index())
    col = ward_colors[ward]
    fig.add_trace(go.Scatter(
        x=dfw['date'], y=dfw['count_0'],
        mode='lines+markers', line=dict(color=col),
        error_y=dict(type='data', array=np.sqrt(dfw['count_0']), visible=True,
                     color=col, thickness=1, width=4),
        name=f'ward {ward} count_0', legend=leg_ref9, showlegend=True,
    ), row=11, col=1)
    fig.add_trace(go.Scatter(
        x=dfw['date'] + pd.DateOffset(months=1), y=dfw['count_1_pred'],
        mode='lines+markers', line=dict(color=col, dash='dot'),
        name=f'ward {ward} +1m', legend=leg_ref9, showlegend=True,
    ), row=11, col=1)
    fig.add_trace(go.Scatter(
        x=dfw['date'] + pd.DateOffset(months=2), y=dfw['count_2_pred'],
        mode='lines+markers', line=dict(color=col, dash='dash'),
        name=f'ward {ward} +2m', legend=leg_ref9, showlegend=True,
    ), row=11, col=1)
    fig.add_trace(go.Scatter(
        x=dfw['date'] + pd.DateOffset(months=3), y=dfw['count_3_pred'],
        mode='lines+markers', line=dict(color=col, dash='dashdot'),
        name=f'ward {ward} +3m', legend=leg_ref9, showlegend=True,
    ), row=11, col=1)
    fig.add_trace(go.Scatter(
        x=dfw['date'] + pd.DateOffset(months=4), y=dfw['count_4_pred'],
        mode='lines+markers', line=dict(color=col, dash='longdash'),
        name=f'ward {ward} +4m', legend=leg_ref9, showlegend=True,
    ), row=11, col=1)
fig.update_yaxes(type='log', title_text='Count (log)', row=11, col=1)
fig.update_xaxes(title_text='Date', row=11, col=1)

# ---- Plot 10: THEFT heatmap Chicago March 2026 (row 12) ----
fig.add_trace(go.Densitymap(
    lat=df_theft_march['latitude'],
    lon=df_theft_march['longitude'],
    radius=12,
    colorscale='Hot',
    reversescale=True,
    showscale=False,
    name='THEFT density Mar 2026',
    showlegend=True,
    legend='legend11',
), row=12, col=1)
fig.update_layout(
    map=dict(
        style='open-street-map',
        center=dict(lat=41.85, lon=-87.65),
        zoom=10,
    ),
)

# ---- Position each legend ----
leg_layout = {}
leg_layout.update(leg_lr('legend',    1))   # Plot 1:  TTV ts        (row 1)
leg_layout.update(leg_lr('legend3',   4))   # Plot 2:  scatter s_1   (row 4)
leg_layout.update(leg_lr('legend4',   5))   # Plot 3:  scatter s_2   (row 5)
leg_layout.update(leg_lr('legend5',   6))   # Plot 4:  scatter s_3   (row 6)
leg_layout.update(leg_lr('legend6',   7))   # Plot 5:  scatter s_4   (row 7)
leg_layout.update(leg_ur('legend7',   8))   # Plot 6:  THEFT ts      (row 8)
leg_layout.update(leg_ur('legend8',   9))   # Plot 7:  BURGLARY ts   (row 9)
leg_layout.update(leg_ur('legend9',  10))   # Plot 8:  ARSON ts      (row 10)
leg_layout.update(leg_ur('legend10', 11))   # Plot 9:  ward ts       (row 11)
leg_layout.update(leg_ur('legend11', 12))   # Plot 10: map heatmap   (row 12)
fig.update_layout(**leg_layout)

# ---- Save dashboard ----
fig.write_html('seasonal_model_dashboard.html')
print('\nSaved seasonal_model_dashboard.html')
