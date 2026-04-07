#!/usr/bin/env python3
"""Full data exploration: profile + dashboard for Chicago crime data."""

import math
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 1. Read raw data ──────────────────────────────────────────────────────────
df_raw = pd.read_csv('data/crimes.csv', low_memory=False)
print(f"df_raw records: {len(df_raw):,}")

# ── 2. Clean carriage returns in text columns ─────────────────────────────────
text_cols = df_raw.select_dtypes(include='object').columns.tolist()
df_clean = df_raw.copy()
for col in text_cols:
    df_clean[col] = df_clean[col].str.replace(r'\r\n|\r|\n', ' ', regex=True)
df_clean.to_csv('data/crimes_clean.csv', index=False)
print(f"df_clean records: {len(df_clean):,}")

# ── 3. primary_type counts ────────────────────────────────────────────────────
print("\nprimary_type counts in df_clean:")
print(df_clean['primary_type'].value_counts().to_string())

# ── 4. Filter sensitive categories ────────────────────────────────────────────
exclude = ['CRIMINAL SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'SEX OFFENSE', 'PROSTITUTION']
df_filtered = df_clean[~df_clean['primary_type'].isin(exclude)].copy()
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
df_filtered.to_csv('data/crimes_filtered.csv', index=False)
print(f"\ndf_filtered records: {len(df_filtered):,}")

# ── 5. Profile df_filtered ────────────────────────────────────────────────────
print("\n── df_filtered profile ──────────────────────────────────────────")
print(f"Shape: {df_filtered.shape}")
print(f"Date range: {df_filtered['date'].min()} → {df_filtered['date'].max()}")
print(f"\nNull counts:")
print(df_filtered.isnull().sum().to_string())
print(f"\nNumeric summary:")
print(df_filtered.describe().to_string())
print(f"\ndtypes:")
print(df_filtered.dtypes.to_string())

# ── 6. Dashboard ──────────────────────────────────────────────────────────────
daily = df_filtered.groupby(df_filtered['date'].dt.date).size().reset_index(name='count')
daily['date'] = pd.to_datetime(daily['date'])

weekly = df_filtered.groupby(df_filtered['date'].dt.to_period('W')).size().reset_index(name='count')
weekly['date'] = weekly['date'].apply(lambda p: p.start_time)

monthly = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size().reset_index(name='count')
monthly['date'] = monthly['date'].apply(lambda p: p.start_time)

primary_counts = df_filtered['primary_type'].value_counts().sort_values(ascending=False)

ward_counts = (df_filtered.dropna(subset=['ward'])
               .assign(ward=lambda d: d['ward'].astype(int))
               .groupby('ward').size()
               .reset_index(name='count')
               .sort_values('count', ascending=False))

# Plot 6: monthly THEFT counts by ward
theft = df_filtered[df_filtered['primary_type'] == 'THEFT'].dropna(subset=['ward']).copy()
theft['ward'] = theft['ward'].astype(int)
theft['month'] = theft['date'].dt.to_period('M').apply(lambda p: p.start_time)
theft_by_ward = theft.groupby(['month', 'ward']).size().reset_index(name='count')
wards = sorted(theft_by_ward['ward'].unique())

# Plot 7: 10000 random records
sample = df_filtered.dropna(subset=['latitude', 'longitude', 'ward']).sample(10000, random_state=42).copy()
sample['ward'] = sample['ward'].astype(int)
sample['neg_longitude'] = -sample['longitude']

def make_colors(items):
    n = len(items)
    return [f'hsl({int(i * 360 / n)},70%,50%)' for i in range(n)]

fig = make_subplots(
    rows=7, cols=1,
    subplot_titles=(
        'Daily Crime Count',
        'Weekly Crime Count',
        'Monthly Crime Count',
        'Crime Count by Primary Type (log scale)',
        'Crime Count by Ward',
        'Monthly THEFT Count by Ward (log scale, y: 10–700)',
        'Lat vs −Lon — 10,000 Random Records (colored by Ward)',
    ),
    vertical_spacing=0.055,
    row_heights=[3, 3, 3, 5, 5, 9, 9],
)

# Plot 1 — daily
fig.add_trace(go.Scatter(
    x=daily['date'], y=daily['count'],
    mode='lines', line=dict(color='steelblue', width=1),
    showlegend=False,
), row=1, col=1)

# Plot 2 — weekly
fig.add_trace(go.Scatter(
    x=weekly['date'], y=weekly['count'],
    mode='lines', line=dict(color='darkorange', width=2),
    showlegend=False,
), row=2, col=1)

# Plot 3 — monthly
fig.add_trace(go.Scatter(
    x=monthly['date'], y=monthly['count'],
    mode='lines+markers', line=dict(color='seagreen', width=2),
    marker=dict(size=6),
    showlegend=False,
), row=3, col=1)

# Plot 4 — primary_type bar, log y
fig.add_trace(go.Bar(
    x=primary_counts.index,
    y=primary_counts.values,
    marker_color='mediumpurple',
    showlegend=False,
), row=4, col=1)
fig.update_yaxes(type='log', title_text='Count (log)', row=4, col=1)
fig.update_xaxes(tickangle=40, row=4, col=1)

# Plot 5 — ward bar
fig.add_trace(go.Bar(
    x=ward_counts['ward'].astype(str),
    y=ward_counts['count'],
    marker_color='coral',
    showlegend=False,
), row=5, col=1)
fig.update_yaxes(title_text='Count', row=5, col=1)
fig.update_xaxes(title_text='Ward', tickangle=0, row=5, col=1)

# Plot 6 — THEFT by ward, log y, no legend
colors6 = make_colors(wards)
for i, ward in enumerate(wards):
    wd = theft_by_ward[theft_by_ward['ward'] == ward]
    fig.add_trace(go.Scatter(
        x=wd['month'], y=wd['count'],
        mode='lines',
        name=f'Ward {ward}',
        line=dict(color=colors6[i], width=1.2),
        showlegend=False,
    ), row=6, col=1)
fig.update_yaxes(
    type='log',
    range=[math.log10(10), math.log10(700)],
    title_text='# THEFT (log)',
    row=6, col=1,
)
fig.update_xaxes(title_text='Date', row=6, col=1)

# Plot 7 — scatter lat vs -lon, colored by ward
ward_vals = sorted(sample['ward'].unique())
colors7 = make_colors(ward_vals)
for i, ward in enumerate(ward_vals):
    s = sample[sample['ward'] == ward]
    fig.add_trace(go.Scatter(
        x=s['neg_longitude'], y=s['latitude'],
        mode='markers',
        name=f'Ward {ward}',
        marker=dict(color=colors7[i], size=4, opacity=0.65),
        showlegend=False,
        hovertemplate=f'Ward {ward}<br>Lat: %{{y:.4f}}<br>−Lon: %{{x:.4f}}<extra></extra>',
    ), row=7, col=1)
fig.update_xaxes(title_text='−Longitude', range=[87.85, 87.5], row=7, col=1)
fig.update_yaxes(title_text='Latitude', range=[41.65, 42.05], row=7, col=1)
fig.update_layout(yaxis7=dict(
    scaleanchor='x7',
    scaleratio=1 / math.cos(math.radians(41.85)),
))

# shared axis labels for plots 1–3
for row in [1, 2, 3]:
    fig.update_yaxes(title_text='# Records', row=row, col=1)
    fig.update_xaxes(title_text='Date', row=row, col=1)

fig.update_layout(
    title=dict(text='Chicago Crimes — Data Exploration Dashboard', font=dict(size=22)),
    width=1100,
    height=5400,
    template='plotly_white',
    showlegend=False,
)

fig.write_html('data_exploration.html', include_plotlyjs='cdn')
print('\nSaved data_exploration.html')
