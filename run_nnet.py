"""Train a simple MLP neural network on Chicago crime data and generate predictions + dashboard."""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

np.random.seed(42)
tf.random.set_seed(42)

# ---- Section 1: load data ----
df_monthly = pd.read_csv('data/crimes_monthly.csv')
cols = ['date', 'year', 'month', 'ward', 'primary_type', 'delta_count',
        'count_0', 'count_1', 'count_2', 'count_3', 'count_4']
df_train = df_monthly[df_monthly['TTV'] == 'train'][cols].copy()
df_test = df_monthly[df_monthly['TTV'] == 'test'][cols].copy()
df_validate = df_monthly[df_monthly['TTV'] == 'validate'][cols].copy()
df_forecast = df_monthly[df_monthly['TTV'] == 'forecast'][cols].copy()

print(f"df_train:    {len(df_train)}")
print(f"df_test:     {len(df_test)}")
print(f"df_validate: {len(df_validate)}")
print(f"df_forecast: {len(df_forecast)}")
print("\n5 random records from df_train:")
print(df_train.sample(5, random_state=1))

# ---- Section 2: train MLP ----
input_cols = ['year', 'month', 'ward', 'primary_type', 'delta_count', 'count_0']
output_cols = ['count_1', 'count_2', 'count_3', 'count_4']

primary_types = sorted(df_monthly['primary_type'].unique().tolist())
pt_to_idx = {pt: i for i, pt in enumerate(primary_types)}

def featurize(df):
    return np.column_stack([
        df['year'].values.astype(float),
        df['month'].values.astype(float),
        df['ward'].values.astype(float),
        df['primary_type'].map(pt_to_idx).values.astype(float),
        df['delta_count'].values.astype(float),
        df['count_0'].values.astype(float),
    ])

X_train = featurize(df_train)
y_train = df_train[output_cols].values.astype(float)
X_test = featurize(df_test)
y_test = df_test[output_cols].values.astype(float)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)
y_test_s = y_scaler.transform(y_test)

# 6 inputs, 4 outputs: use 32-neuron hidden layer (roughly 2*(in+out))
nnet = keras.Sequential([
    keras.layers.Input(shape=(6,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='linear'),
])
nnet.compile(optimizer='adam', loss='mse', metrics=['mae'])
nnet.fit(X_train_s, y_train_s,
         validation_data=(X_test_s, y_test_s),
         epochs=40, batch_size=128, verbose=2)

os.makedirs('models', exist_ok=True)
nnet.save('models/nnet.keras')
np.savez('models/nnet_preproc.npz',
         x_mean=x_scaler.mean_, x_scale=x_scaler.scale_,
         y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
         primary_types=np.array(primary_types))
print("\nSaved nnet to models/nnet.keras")

# ---- Section 3: predictions ----
def predict(df):
    X = featurize(df)
    Xs = x_scaler.transform(X)
    ys = nnet.predict(Xs, verbose=0)
    y = y_scaler.inverse_transform(ys)
    out = df.copy()
    for i, c in enumerate(output_cols):
        out[f'{c}_pred'] = y[:, i]
    return out

df_test = predict(df_test)
df_validate = predict(df_validate)
df_forecast = predict(df_forecast)

print("\n5 random records from df_validate:")
print(df_validate.sample(5, random_state=1))

# ---- Section 4: dashboard ----
df_monthly['date'] = pd.to_datetime(df_monthly['date'])
df_validate['date'] = pd.to_datetime(df_validate['date'])

# Plot 1
g = df_monthly.groupby(['date', 'TTV'])['count_0'].sum().reset_index(name='total_count')
train_test = g[g['TTV'].isin(['train', 'test'])].groupby('date')['total_count'].sum().reset_index()
val = g[g['TTV'] == 'validate']
fc = g[g['TTV'] == 'forecast']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=train_test['date'], y=train_test['total_count'],
                          mode='lines+markers', name='train+test', line=dict(color='steelblue')))
fig1.add_trace(go.Scatter(x=val['date'], y=val['total_count'],
                          mode='lines+markers', name='validate', line=dict(color='orange')))
fig1.add_trace(go.Scatter(x=fc['date'], y=fc['total_count'],
                          mode='lines+markers', name='forecast', line=dict(color='green')))
fig1.update_layout(title='Plot 1: Total count_0 vs date by TTV',
                   xaxis_title='date', yaxis_title='total count_0',
                   showlegend=True, legend=dict(x=1.02, y=1),
                   margin=dict(l=40, r=40, t=50, b=40))

# Table 1
rows = []
for c in output_cols:
    mask = df_validate[c].notna() & df_validate[f'{c}_pred'].notna()
    y_true = df_validate.loc[mask, c].values
    y_pred = df_validate.loc[mask, f'{c}_pred'].values
    rows.append({
        'target': c,
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R2': round(r2_score(y_true, y_pred), 4),
    })
scores = pd.DataFrame(rows)
print("\nValidation scores:")
print(scores)
table_html = scores.to_html(index=False, border=1, classes='score-table')

# Plot 2: THEFT
theft = df_validate[df_validate['primary_type'] == 'THEFT']
theft_g = theft.groupby('date').agg(
    count_0=('count_0', 'sum'),
    count_1_pred=('count_1_pred', 'sum'),
    count_2_pred=('count_2_pred', 'sum'),
    count_3_pred=('count_3_pred', 'sum'),
    count_4_pred=('count_4_pred', 'sum'),
).reset_index().sort_values('date')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=theft_g['date'], y=theft_g['count_0'],
    mode='lines+markers', name='count_0 (THEFT)',
    line=dict(color='blue'),
    error_y=dict(type='data', array=np.sqrt(theft_g['count_0'].clip(lower=0)), visible=True),
))
for k, color in zip([1, 2, 3, 4], ['red', 'green', 'orange', 'purple']):
    fig2.add_trace(go.Scatter(
        x=theft_g['date'] + pd.DateOffset(months=k),
        y=theft_g[f'count_{k}_pred'],
        mode='lines+markers', name=f'count_{k}_pred (+{k}mo)', line=dict(color=color),
    ))
fig2.update_layout(title='Plot 2: THEFT count_0 and predictions vs date',
                   xaxis_title='date', yaxis_title='sum count',
                   showlegend=True, legend=dict(x=1.02, y=1),
                   margin=dict(l=40, r=40, t=50, b=40))

# Plot 3: wards 27/29/38, log y
fig3 = go.Figure()
ward_colors = {27: 'red', 29: 'blue', 38: 'green'}
for w, color in ward_colors.items():
    sub = df_validate[df_validate['ward'] == w]
    if len(sub) == 0:
        continue
    g_w = sub.groupby('date').agg(
        count_0=('count_0', 'sum'),
        count_1_pred=('count_1_pred', 'sum'),
        count_2_pred=('count_2_pred', 'sum'),
        count_3_pred=('count_3_pred', 'sum'),
        count_4_pred=('count_4_pred', 'sum'),
    ).reset_index().sort_values('date')
    fig3.add_trace(go.Scatter(
        x=g_w['date'], y=g_w['count_0'],
        mode='lines+markers', name=f'ward {w} count_0',
        line=dict(color=color),
        error_y=dict(type='data', array=np.sqrt(g_w['count_0'].clip(lower=0)), visible=True),
    ))
    for k, dash in zip([1, 2, 3, 4], ['dot', 'dash', 'longdash', 'dashdot']):
        fig3.add_trace(go.Scatter(
            x=g_w['date'] + pd.DateOffset(months=k),
            y=g_w[f'count_{k}_pred'],
            mode='lines+markers',
            name=f'ward {w} count_{k}_pred (+{k}mo)',
            line=dict(color=color, dash=dash),
        ))
fig3.update_layout(title='Plot 3: Wards 27, 29, 38 — count_0 with predictions',
                   xaxis_title='date', yaxis_title='sum count (log)',
                   yaxis_type='log',
                   showlegend=True,
                   legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top'),
                   margin=dict(l=40, r=40, t=50, b=40))

html_parts = [
    '<html><head><meta charset="utf-8"><title>nnet dashboard</title>',
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
    '<style>',
    'body{font-family:Arial,sans-serif;margin:10px;padding:0;}',
    'h1{margin:4px 0;padding:0;}',
    'h3{margin:0 0 6px 0;padding:0;}',
    '.block{margin:0 0 20px 0;padding:0;}',
    '.score-table{border-collapse:collapse;margin:0;padding:0;}',
    '.score-table th,.score-table td{padding:4px 10px;border:1px solid #888;}',
    '</style></head><body>',
    '<h1>Neural Net Model Dashboard</h1>',
    '<div class="block">' + fig1.to_html(full_html=False, include_plotlyjs=False) + '</div>',
    '<div class="block"><h3>Table 1: Validation scores</h3>' + table_html + '</div>',
    '<div class="block">' + fig2.to_html(full_html=False, include_plotlyjs=False) + '</div>',
    '<div class="block">' + fig3.to_html(full_html=False, include_plotlyjs=False) + '</div>',
    '</body></html>',
]
with open('nnet_dashboard.html', 'w') as f:
    f.write('\n'.join(html_parts))
print("\nWrote nnet_dashboard.html")
