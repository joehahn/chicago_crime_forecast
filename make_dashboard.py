#!/usr/bin/env python3
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- build df_filtered ---
df_raw = pd.read_csv('data/crimes_clean.csv', low_memory=False)
exclude = ['CRIMINAL SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'SEX OFFENSE', 'PROSTITUTION']
df_filtered = df_raw[~df_raw['primary_type'].isin(exclude)].copy()
df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# --- aggregations ---
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

# plot 6: monthly THEFT counts by ward
theft = df_filtered[df_filtered['primary_type'] == 'THEFT'].dropna(subset=['ward']).copy()
theft['ward'] = theft['ward'].astype(int)
theft['month'] = theft['date'].dt.to_period('M').apply(lambda p: p.start_time)
theft_by_ward = theft.groupby(['month', 'ward']).size().reset_index(name='count')
wards = sorted(theft_by_ward['ward'].unique())

# plot 7: 10000 random records
sample = df_filtered.dropna(subset=['latitude', 'longitude', 'ward']).sample(10000, random_state=42).copy()
sample['ward'] = sample['ward'].astype(int)
sample['neg_longitude'] = -sample['longitude']

# --- color helpers ---
def make_colors(items):
    n = len(items)
    return [f'hsl({int(i * 360 / n)},70%,50%)' for i in range(n)]

# --- build figure (7 rows) ---
fig = make_subplots(
    rows=7, cols=1,
    subplot_titles=(
        'Daily Crime Count',
        'Weekly Crime Count',
        'Monthly Crime Count',
        'Crime Count by Primary Type (log scale)',
        'Crime Count by Ward (linear scale)',
        'Monthly THEFT Count by Ward (log scale, y: 10–700)',
        'Lat vs −Lon Scatter — 10 000 Random Records (colored by Ward)',
    ),
    vertical_spacing=0.07,
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

# Plot 4 — primary_type vertical bar, log y
fig.add_trace(go.Bar(
    x=primary_counts.index,
    y=primary_counts.values,
    marker_color='mediumpurple',
    showlegend=False,
), row=4, col=1)
fig.update_yaxes(type='log', title_text='Count (log)', row=4, col=1)
fig.update_xaxes(tickangle=40, row=4, col=1)

# Plot 5 — ward vertical bar, linear y
fig.add_trace(go.Bar(
    x=ward_counts['ward'].astype(str),
    y=ward_counts['count'],
    marker_color='coral',
    showlegend=False,
), row=5, col=1)
fig.update_yaxes(title_text='Count', row=5, col=1)
fig.update_xaxes(title_text='Ward', tickangle=0, row=5, col=1)

# Plot 6 — THEFT by ward timeseries, log y, no legend
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
    range=[1, 2.845],   # log10(10)=1, log10(700)≈2.845
    title_text='# THEFT (log)',
    row=6, col=1,
)
fig.update_xaxes(title_text='Date', row=6, col=1)

# Plot 7 — scatter lat vs -lon, colored by ward, no legend
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
# equal physical aspect ratio: 1 deg lat ≈ 1/cos(lat) deg lon at Chicago latitude
fig.update_layout(yaxis7=dict(
    scaleanchor='x7',
    scaleratio=1 / math.cos(math.radians(41.85)),
))

# shared labels for plots 1-3
for row in [1, 2, 3]:
    fig.update_yaxes(title_text='# Records', row=row, col=1)
    fig.update_xaxes(title_text='Date',      row=row, col=1)

fig.update_layout(
    title=dict(text='Chicago Crimes — Data Exploration Dashboard', font=dict(size=22)),
    width=1100,
    height=5200,
    template='plotly_white',
    showlegend=False,
)

fig.write_html('data_exploration.html', include_plotlyjs='cdn')
print('Saved data_exploration.html')
