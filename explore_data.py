#!/usr/bin/env python3
# explore_data.py
# Profile the Chicago crimes dataset and produce an HTML dashboard of exploratory plots.

import os
from math import cos, radians

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the cleaned & filtered crimes dataset saved by get_data.py
df_filtered = pd.read_csv("data/crimes.csv", low_memory=False)
print(f"Records in df_filtered: {len(df_filtered):,}")

# Profile df_filtered: shape, dtypes, nulls, uniques, numeric stats, head
print("\n--- Shape ---")
print(df_filtered.shape)

print("\n--- Dtypes ---")
print(df_filtered.dtypes.to_string())

print("\n--- Null counts ---")
nulls = df_filtered.isna().sum()
print(nulls[nulls > 0].sort_values(ascending=False).to_string() or "(none)")

print("\n--- Unique counts per column ---")
print(df_filtered.nunique().sort_values(ascending=False).to_string())

print("\n--- Numeric summary ---")
print(df_filtered.describe(include="number").to_string())

print("\n--- Head (3 rows) ---")
print(df_filtered.head(3).to_string())


# ---------- Build exploratory dashboard ----------
print("\nBuilding dashboard plots...")
df_filtered["date"] = pd.to_datetime(df_filtered["date"])
os.makedirs("docs/img", exist_ok=True)

# Plot 1: daily record count (Scattergl because >1000 points)
daily = df_filtered.groupby(df_filtered["date"].dt.floor("D")).size()
fig1 = go.Figure(go.Scattergl(x=daily.index, y=daily.values, mode="lines"))
fig1.update_layout(
    title="Daily crime count",
    xaxis_title="Date",
    yaxis_title="Count",
    height=350,
    margin=dict(l=60, r=20, t=50, b=50),
)

# Plot 2: weekly record count
weekly = df_filtered.set_index("date").resample("W").size()
fig2 = go.Figure(go.Scatter(x=weekly.index, y=weekly.values, mode="lines"))
fig2.update_layout(
    title="Weekly crime count",
    xaxis_title="Date",
    yaxis_title="Count",
    height=350,
    margin=dict(l=60, r=20, t=50, b=50),
)

# Plot 3: monthly record count
monthly = df_filtered.set_index("date").resample("MS").size()
fig3 = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, mode="lines"))
fig3.update_layout(
    title="Monthly crime count",
    xaxis_title="Date",
    yaxis_title="Count",
    height=350,
    margin=dict(l=60, r=20, t=50, b=50),
)

# Plot 4: primary_type counts, descending, log y
ptype_counts = df_filtered["primary_type"].value_counts()
fig4 = go.Figure(go.Bar(x=ptype_counts.index, y=ptype_counts.values))
fig4.update_layout(
    title="primary_type counts",
    xaxis_title="primary_type",
    yaxis_title="Count (log)",
    yaxis_type="log",
    height=500,
    margin=dict(l=60, r=20, t=50, b=140),
)

# Plot 5: ward counts, descending
ward_counts = df_filtered["ward"].value_counts()
fig5 = go.Figure(
    go.Bar(x=ward_counts.index.astype(int).astype(str), y=ward_counts.values)
)
fig5.update_layout(
    title="Ward counts",
    xaxis_title="ward",
    yaxis_title="Count",
    height=400,
    xaxis=dict(type="category"),
    margin=dict(l=60, r=20, t=50, b=60),
)

# Plot 6 (static PNG): monthly THEFT count per ward
theft = df_filtered[df_filtered["primary_type"] == "THEFT"].copy()
theft["month"] = theft["date"].dt.to_period("M").dt.to_timestamp()
theft_by_ward = theft.groupby(["month", "ward"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
for ward in theft_by_ward.columns:
    ax.plot(theft_by_ward.index, theft_by_ward[ward], alpha=0.6)
ax.set_yscale("log")
ax.set_ylim(10, 1100)
ax.set_xlabel("Date")
ax.set_ylabel("THEFT count (log)")
ax.set_title("Monthly THEFT count per ward")
plt.tight_layout()
plt.savefig("docs/img/theft_per_ward.png", dpi=120)
plt.close()

# Plot 7 (static PNG): geographic scatter of 10k random records, colored by ward
geo = df_filtered.dropna(subset=["latitude", "longitude", "ward"])
sample = geo.sample(n=10_000, random_state=42)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    -sample["longitude"],
    sample["latitude"],
    c=sample["ward"],
    cmap="tab20",
    s=8,
    alpha=0.8,
)
ax.set_xlim(87.85, 87.5)  # inverted: 87.85 on left, 87.5 on right
ax.set_ylim(41.65, 42.05)
# Aspect ratio compensates for longitude foreshortening at Chicago's latitude
ax.set_aspect(1 / cos(radians(41.85)))
ax.set_xlabel("-longitude")
ax.set_ylabel("latitude")
ax.set_title("Chicago crime locations (10k random sample, colored by ward)")
plt.tight_layout()
plt.savefig("docs/img/geo_scatter.png", dpi=120)
plt.close()

# Assemble dashboard: adjacent siblings get exactly 10px of vertical space between them
css = """
body { margin: 0; padding: 0; font-family: sans-serif; }
.plot { margin: 0; padding: 0; }
.plot + .plot { margin-top: 10px; }
img { display: block; max-width: 100%; height: auto; }
"""

html_parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Chicago crime — data exploration</title>",
    f"<style>{css}</style></head><body>",
]

# First plotly figure loads plotly.js from CDN; the rest reuse that runtime
html_parts.append(
    f"<div class='plot'>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
)
for fig in (fig2, fig3, fig4, fig5):
    html_parts.append(
        f"<div class='plot'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
    )
html_parts.append("<div class='plot'><img src='img/theft_per_ward.png' alt='THEFT per ward'></div>")
html_parts.append("<div class='plot'><img src='img/geo_scatter.png' alt='Chicago geo scatter'></div>")
html_parts.append("</body></html>")

with open("docs/data_exploration.html", "w") as f:
    f.write("".join(html_parts))
print("Saved docs/data_exploration.html")
