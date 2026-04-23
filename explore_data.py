"""
explore_data.py

Reads the Chicago crimes dataset, profiles it, and produces an HTML
dashboard of exploratory plots at docs/data_exploration.html.

Rendering rules:
  - plotly figures share a single CDN-hosted plotly.js runtime
    (first figure: include_plotlyjs='cdn'; rest: include_plotlyjs=False)
  - scatter plots with >1,000 points use Scattergl (WebGL)
  - plots flagged as "static PNG" are rendered with matplotlib to
    docs/img/*.png at dpi=120 and embedded as <img> tags.
"""

import os
from math import cos, radians

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Read data/crimes.csv.
# ---------------------------------------------------------------------------

df_filtered = pd.read_csv("data/crimes.csv", low_memory=False)
print(f"df_filtered has {len(df_filtered):,} records")


# ---------------------------------------------------------------------------
# Profile df_filtered.
# ---------------------------------------------------------------------------

print("\nshape:", df_filtered.shape)
print("\ndtypes:")
print(df_filtered.dtypes)
print("\ndescribe (all):")
print(df_filtered.describe(include="all").to_string())
print("\nnull counts:")
print(df_filtered.isna().sum())
print("\nhead:")
print(df_filtered.head().to_string())


# ---------------------------------------------------------------------------
# Derive a datetime column used by the timeseries plots.
# ---------------------------------------------------------------------------

df_filtered["date"] = pd.to_datetime(df_filtered["date"], errors="coerce")

# daily / weekly / monthly counts for Plots 1-3
daily = df_filtered.set_index("date").resample("D").size().rename("count").reset_index()
weekly = df_filtered.set_index("date").resample("W").size().rename("count").reset_index()
monthly = df_filtered.set_index("date").resample("MS").size().rename("count").reset_index()


# ---------------------------------------------------------------------------
# Build interactive plotly figures (Plots 1-5).
# ---------------------------------------------------------------------------

# Plot 1 — daily timeseries
fig1 = go.Figure()
fig1.add_trace(go.Scattergl(x=daily["date"], y=daily["count"], mode="lines", name="daily"))
fig1.update_layout(
    title="Plot 1 — Daily crime count",
    xaxis_title="date",
    yaxis_title="records per day",
    height=380,
    margin=dict(l=60, r=30, t=50, b=50),
)

# Plot 2 — weekly timeseries
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=weekly["date"], y=weekly["count"], mode="lines", name="weekly"))
fig2.update_layout(
    title="Plot 2 — Weekly crime count",
    xaxis_title="date",
    yaxis_title="records per week",
    height=380,
    margin=dict(l=60, r=30, t=50, b=50),
)

# Plot 3 — monthly timeseries
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=monthly["date"], y=monthly["count"], mode="lines+markers", name="monthly"))
fig3.update_layout(
    title="Plot 3 — Monthly crime count",
    xaxis_title="date",
    yaxis_title="records per month",
    height=380,
    margin=dict(l=60, r=30, t=50, b=50),
)

# Plot 4 — primary_type bar chart (descending, log y)
pt_counts = df_filtered["primary_type"].value_counts()
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=pt_counts.index, y=pt_counts.values))
fig4.update_layout(
    title="Plot 4 — Count by primary_type (log y)",
    xaxis_title="primary_type",
    yaxis_title="records (log)",
    yaxis_type="log",
    height=500,
    margin=dict(l=60, r=30, t=50, b=150),
)

# Plot 5 — ward bar chart (descending)
ward_counts = df_filtered["ward"].value_counts()
fig5 = go.Figure()
fig5.add_trace(go.Bar(x=ward_counts.index.astype(str), y=ward_counts.values))
fig5.update_layout(
    title="Plot 5 — Count by ward",
    xaxis_title="ward",
    yaxis_title="records",
    xaxis_type="category",
    height=420,
    margin=dict(l=60, r=30, t=50, b=60),
)


# ---------------------------------------------------------------------------
# Plot 6 (static PNG) — THEFT timeseries per ward.
# ---------------------------------------------------------------------------

os.makedirs("docs/img", exist_ok=True)

theft = df_filtered[df_filtered["primary_type"] == "THEFT"].copy()
# monthly count per ward
theft_monthly = (
    theft.set_index("date")
    .groupby("ward")
    .resample("MS")
    .size()
    .rename("count")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))
for ward, g in theft_monthly.groupby("ward"):
    ax.plot(g["date"], g["count"], alpha=0.6, linewidth=1.2)
ax.set_yscale("log")
ax.set_ylim(10, 1100)
ax.set_xlabel("date")
ax.set_ylabel("monthly THEFT count (log)")
ax.set_title("Plot 6 — Monthly THEFT count per ward")
fig.tight_layout()
theft_path = "docs/img/theft_per_ward.png"
fig.savefig(theft_path, dpi=120)
plt.close(fig)
print(f"\nsaved {theft_path}")


# ---------------------------------------------------------------------------
# Plot 7 (static PNG) — geo scatter of 10,000 random records.
# ---------------------------------------------------------------------------

geo = df_filtered.dropna(subset=["latitude", "longitude", "ward"]).copy()
sample = geo.sample(n=min(10_000, len(geo)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(
    -sample["longitude"].astype(float),
    sample["latitude"].astype(float),
    c=sample["ward"].astype(int),
    cmap="tab20",
    s=8,
    alpha=0.8,
)
ax.set_xlim(87.85, 87.5)  # note: east-is-right, so 87.85 on the left
ax.set_ylim(41.65, 42.05)
ax.set_aspect(1.0 / cos(radians(41.85)))
ax.set_xlabel("-longitude")
ax.set_ylabel("latitude")
ax.set_title("Plot 7 — Geo scatter (10k random records, colored by ward)")
fig.tight_layout()
geo_path = "docs/img/geo_scatter.png"
fig.savefig(geo_path, dpi=120)
plt.close(fig)
print(f"saved {geo_path}")


# ---------------------------------------------------------------------------
# Assemble the HTML dashboard. Every adjacent pair of elements has exactly
# 10px of vertical spacing (margin-bottom on the wrapper, last-child none).
# ---------------------------------------------------------------------------

html_parts = []
html_parts.append(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chicago Crime — Data Exploration</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; margin: 16px; }
  h1 { margin: 0 0 10px 0; }
  .block { margin: 0 0 10px 0; }
  .block:last-child { margin-bottom: 0; }
  img { display: block; max-width: 100%; height: auto; }
</style>
</head>
<body>
<div class="block"><h1>Chicago Crime — Data Exploration</h1></div>
"""
)

# Plot 1 includes plotly.js from CDN; subsequent plots reuse it.
html_parts.append(
    '<div class="block">'
    + fig1.to_html(full_html=False, include_plotlyjs="cdn")
    + "</div>"
)
html_parts.append(
    '<div class="block">'
    + fig2.to_html(full_html=False, include_plotlyjs=False)
    + "</div>"
)
html_parts.append(
    '<div class="block">'
    + fig3.to_html(full_html=False, include_plotlyjs=False)
    + "</div>"
)
html_parts.append(
    '<div class="block">'
    + fig4.to_html(full_html=False, include_plotlyjs=False)
    + "</div>"
)
html_parts.append(
    '<div class="block">'
    + fig5.to_html(full_html=False, include_plotlyjs=False)
    + "</div>"
)

# Plots 6 and 7 are static PNGs embedded via <img> tags.
html_parts.append('<div class="block"><img src="img/theft_per_ward.png" alt="Plot 6 — THEFT per ward"></div>')
html_parts.append('<div class="block"><img src="img/geo_scatter.png" alt="Plot 7 — geo scatter"></div>')

html_parts.append("</body></html>")

out_path = "docs/data_exploration.html"
with open(out_path, "w") as f:
    f.write("\n".join(html_parts))
print(f"saved {out_path}")
