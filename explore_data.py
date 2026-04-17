#!/usr/bin/env python3
"""Profile the Chicago crimes dataset and render an HTML exploration dashboard.

Reads  : data/crimes.csv        (produced by get_data.py)
Writes : data_exploration.html
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED = 42
SCATTER_SAMPLE_SIZE = 10_000

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "data" / "crimes.csv"
OUTPUT_PATH = ROOT / "data_exploration.html"


# 1. Load & profile.
print(f"Loading {INPUT_PATH} ...")
df_filtered = pd.read_csv(INPUT_PATH)
df_filtered["date"] = pd.to_datetime(df_filtered["date"])
print(f"Records in df_filtered: {len(df_filtered):,}")

print("\n--- Profile df_filtered ---")
print(df_filtered.dtypes)
print()
print(df_filtered.describe(include="all"))

# 2. Aggregations for timeseries plots.
daily = df_filtered.groupby(df_filtered["date"].dt.date).size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])

weekly = df_filtered.groupby(df_filtered["date"].dt.to_period("W")).size().reset_index(name="count")
weekly["date"] = weekly["date"].apply(lambda p: p.start_time)

monthly = df_filtered.groupby(df_filtered["date"].dt.to_period("M")).size().reset_index(name="count")
monthly["date"] = monthly["date"].apply(lambda p: p.start_time)

# 3. Aggregations for bar charts.
type_counts = df_filtered["primary_type"].value_counts().sort_values(ascending=False)
ward_counts = df_filtered["ward"].dropna().astype(int).value_counts().sort_values(ascending=False)

# 4. THEFT by ward, monthly.
theft = df_filtered[df_filtered["primary_type"] == "THEFT"].copy()
theft["month"] = theft["date"].dt.to_period("M").apply(lambda p: p.start_time)
theft_by_ward = theft.groupby(["month", "ward"]).size().reset_index(name="count")
wards = sorted(theft_by_ward["ward"].dropna().unique())

# 5. Geo scatter sample.
scatter_sample = (
    df_filtered.dropna(subset=["latitude", "longitude", "ward"])
    .sample(SCATTER_SAMPLE_SIZE, random_state=SEED)
    .copy()
)
scatter_sample["neg_longitude"] = -scatter_sample["longitude"]

# 6. Build dashboard.
fig = make_subplots(
    rows=7,
    cols=1,
    subplot_titles=(
        "Plot 1 — Daily crime count",
        "Plot 2 — Weekly crime count",
        "Plot 3 — Monthly crime count",
        "Plot 4 — Count by primary_type (log y)",
        "Plot 5 — Count by ward",
        "Plot 6 — Monthly THEFT count by ward (log y)",
        f"Plot 7 — {SCATTER_SAMPLE_SIZE:,} random locations (latitude vs. -longitude), colored by ward",
    ),
    vertical_spacing=0.07,
    row_heights=[1, 1, 1, 1.4, 1.4, 1, 1.6],
)

# Plot 1
fig.add_trace(
    go.Scatter(x=daily["date"], y=daily["count"], mode="lines",
               line=dict(color="steelblue", width=1), showlegend=False),
    row=1, col=1,
)

# Plot 2
fig.add_trace(
    go.Scatter(x=weekly["date"], y=weekly["count"], mode="lines",
               line=dict(color="darkorange", width=1.5), showlegend=False),
    row=2, col=1,
)

# Plot 3
fig.add_trace(
    go.Scatter(x=monthly["date"], y=monthly["count"], mode="lines+markers",
               line=dict(color="green", width=2), showlegend=False),
    row=3, col=1,
)

# Plot 4 — bar of primary_type, log y
fig.add_trace(
    go.Bar(x=type_counts.index, y=type_counts.values,
           marker_color="mediumpurple", showlegend=False),
    row=4, col=1,
)
fig.update_yaxes(type="log", title_text="count (log)", row=4, col=1)
fig.update_xaxes(tickangle=45, row=4, col=1)

# Plot 5 — bar of ward
fig.add_trace(
    go.Bar(x=ward_counts.index.astype(str), y=ward_counts.values,
           marker_color="coral", showlegend=False),
    row=5, col=1,
)
fig.update_xaxes(
    title_text="ward", tickangle=90, row=5, col=1,
    categoryorder="array", categoryarray=[str(w) for w in ward_counts.index],
)
fig.update_yaxes(title_text="count", row=5, col=1)

# Plot 6 — THEFT by ward, log y, no legend
colors = [f"hsl({int(i / len(wards) * 360)},70%,50%)" for i in range(len(wards))]
for i, ward in enumerate(wards):
    wd = theft_by_ward[theft_by_ward["ward"] == ward]
    fig.add_trace(
        go.Scatter(x=wd["month"], y=wd["count"], mode="lines",
                   line=dict(color=colors[i], width=1), showlegend=False),
        row=6, col=1,
    )
fig.update_yaxes(type="log", range=[1, 3.041], title_text="count (log)", row=6, col=1)

# Plot 7 — latitude vs. -longitude, colored by ward, geographic aspect
fig.add_trace(
    go.Scatter(
        x=scatter_sample["neg_longitude"],
        y=scatter_sample["latitude"],
        mode="markers",
        marker=dict(
            color=scatter_sample["ward"],
            colorscale="Turbo",
            size=3,
            opacity=0.6,
            colorbar=dict(title="ward", x=1.02, len=0.13, y=0.05),
        ),
        showlegend=False,
    ),
    row=7, col=1,
)
# Reversed range so 87.85 is on the left and 87.5 on the right.
fig.update_xaxes(range=[87.85, 87.5], title_text="-longitude", row=7, col=1)
fig.update_yaxes(
    range=[41.65, 42.05], title_text="latitude", row=7, col=1,
    scaleanchor="x7", scaleratio=1.34,  # 1/cos(41.85°) ≈ geographic aspect
)

fig.update_layout(
    title="Chicago Crime Data — Exploration Dashboard",
    height=5500,
    width=1000,
    showlegend=False,
    template="plotly_white",
    font=dict(size=12),
)

fig.write_html(OUTPUT_PATH)
print(f"\nSaved dashboard to {OUTPUT_PATH}")
