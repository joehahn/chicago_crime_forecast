#!/usr/bin/env python3
# explore_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-9

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df_filtered = pd.read_csv('data/crimes.csv')
df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# ---- Plot 1: Daily counts ----
daily = df_filtered.groupby(df_filtered['date'].dt.date).size().reset_index(name='count')
daily['date'] = pd.to_datetime(daily['date'])

# ---- Plot 2: Weekly counts ----
weekly = df_filtered.groupby(df_filtered['date'].dt.to_period('W')).size().reset_index(name='count')
weekly['date'] = weekly['date'].apply(lambda p: p.start_time)

# ---- Plot 3: Monthly counts ----
monthly = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size().reset_index(name='count')
monthly['date'] = monthly['date'].apply(lambda p: p.start_time)

# ---- Plot 4: Primary type barchart (log scale) ----
type_counts = df_filtered['primary_type'].value_counts().sort_values(ascending=False)

# ---- Plot 5: Ward barchart (decreasing order) ----
ward_counts = df_filtered['ward'].dropna().astype(int).value_counts().sort_values(ascending=False)

# ---- Plot 6: THEFT by ward over time (monthly) ----
theft_df = df_filtered[df_filtered['primary_type'] == 'THEFT'].copy()
theft_df['month'] = theft_df['date'].dt.to_period('M').apply(lambda p: p.start_time)
theft_by_ward = theft_df.groupby(['month', 'ward']).size().reset_index(name='count')
wards = sorted(theft_by_ward['ward'].dropna().unique())

# ---- Plot 7: Scatter lat vs -lon colorized by ward ----
sample = df_filtered.dropna(subset=['latitude', 'longitude', 'ward']).sample(10000, random_state=42)

# Build subplots - 7 rows
fig = make_subplots(
    rows=7, cols=1,
    subplot_titles=(
        'Plot 1: Daily Crime Count',
        'Plot 2: Weekly Crime Count',
        'Plot 3: Monthly Crime Count',
        'Plot 4: Crime Count by Primary Type (log scale)',
        'Plot 5: Crime Count by Ward',
        'Plot 6: Monthly THEFT Count by Ward (log scale)',
        'Plot 7: Crime Locations (10,000 random records, colorized by ward)',
    ),
    vertical_spacing=0.07,
    row_heights=[1, 1, 1, 1.4, 1.4, 1, 1.6],
)

# Plot 1
fig.add_trace(go.Scatter(x=daily['date'], y=daily['count'], mode='lines', name='Daily',
                          line=dict(color='steelblue', width=1)), row=1, col=1)

# Plot 2
fig.add_trace(go.Scatter(x=weekly['date'], y=weekly['count'], mode='lines', name='Weekly',
                          line=dict(color='darkorange', width=1.5)), row=2, col=1)

# Plot 3
fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['count'], mode='lines+markers', name='Monthly',
                          line=dict(color='green', width=2)), row=3, col=1)

# Plot 4 - horizontal bar for readability, but user asked vertical (counts vertical = y axis = counts)
fig.add_trace(go.Bar(
    x=type_counts.index,
    y=type_counts.values,
    name='Primary Type',
    marker_color='mediumpurple',
    showlegend=False,
), row=4, col=1)
fig.update_yaxes(type='log', row=4, col=1, title_text='Count (log)')
fig.update_xaxes(tickangle=45, row=4, col=1)

# Plot 5
fig.add_trace(go.Bar(
    x=ward_counts.index.astype(str),
    y=ward_counts.values,
    name='Ward',
    marker_color='coral',
    showlegend=False,
), row=5, col=1)
fig.update_xaxes(title_text='Ward', row=5, col=1, tickangle=90,
                 categoryorder='array', categoryarray=[str(w) for w in ward_counts.index])
fig.update_yaxes(title_text='Count', row=5, col=1)

# Plot 6 - THEFT by ward (no legend)
colors = [f'hsl({int(i/len(wards)*360)},70%,50%)' for i in range(len(wards))]
for i, ward in enumerate(wards):
    wd = theft_by_ward[theft_by_ward['ward'] == ward]
    fig.add_trace(go.Scatter(
        x=wd['month'], y=wd['count'],
        mode='lines',
        name=f'Ward {int(ward)}',
        line=dict(color=colors[i], width=1),
        showlegend=False,
    ), row=6, col=1)
fig.update_yaxes(type='log', range=[1, 3.041], row=6, col=1, title_text='Count (log)')

# Plot 7 - scatter map
fig.add_trace(go.Scatter(
    x=sample['longitude'],
    y=sample['latitude'],
    mode='markers',
    marker=dict(
        color=sample['ward'],
        colorscale='Turbo',
        size=3,
        opacity=0.6,
        colorbar=dict(title='Ward', x=1.02, len=0.13, y=0.05),
    ),
    name='Location',
    showlegend=False,
), row=7, col=1)
fig.update_xaxes(range=[-87.85, -87.5], title_text='Longitude', row=7, col=1)
fig.update_yaxes(range=[41.65, 42.05], title_text='Latitude', row=7, col=1,
                 scaleanchor='x7', scaleratio=1.3)

# Global layout
fig.update_layout(
    title='Chicago Crime Data Exploration Dashboard',
    height=5500,
    width=1000,
    showlegend=False,
    template='plotly_white',
    font=dict(size=12),
)

fig.write_html('data_exploration.html')
print('Saved data_exploration.html')
