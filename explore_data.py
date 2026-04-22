#!/usr/bin/env python3
"""Profile the Chicago crimes dataset and render an HTML exploration dashboard.

Reads  : data/crimes.csv                    (produced by get_data.py)
Writes : docs/data_exploration.html         (published via GitHub Pages)

Layout: each plot is rendered as an independent Plotly figure, then stacked
in a single HTML page with exactly 20 px of vertical margin between adjacent
plots (enforced via CSS, not via Plotly's subplot `vertical_spacing`).
"""

from math import cos, radians
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

SEED = 42
SCATTER_SAMPLE_SIZE = 10_000
GAP_PX = 10  # exactly 10 px between every pair of adjacent plots
STATIC_DPI = 120

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "data" / "crimes.csv"
OUTPUT_PATH = ROOT / "docs" / "data_exploration.html"
IMG_DIR = ROOT / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load & profile
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_PATH} ...")
df_filtered = pd.read_csv(INPUT_PATH)
df_filtered["date"] = pd.to_datetime(df_filtered["date"])
print(f"Records in df_filtered: {len(df_filtered):,}")

print("\n--- Profile df_filtered ---")
print(df_filtered.dtypes)
print()
print(df_filtered.describe(include="all"))

# ---------------------------------------------------------------------------
# 2. Aggregations
# ---------------------------------------------------------------------------
daily = df_filtered.groupby(df_filtered["date"].dt.date).size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])

weekly = df_filtered.groupby(df_filtered["date"].dt.to_period("W")).size().reset_index(name="count")
weekly["date"] = weekly["date"].apply(lambda p: p.start_time)

monthly = df_filtered.groupby(df_filtered["date"].dt.to_period("M")).size().reset_index(name="count")
monthly["date"] = monthly["date"].apply(lambda p: p.start_time)

type_counts = df_filtered["primary_type"].value_counts().sort_values(ascending=False)
ward_counts = df_filtered["ward"].dropna().astype(int).value_counts().sort_values(ascending=False)

theft = df_filtered[df_filtered["primary_type"] == "THEFT"].copy()
theft["month"] = theft["date"].dt.to_period("M").apply(lambda p: p.start_time)
theft_by_ward = theft.groupby(["month", "ward"]).size().reset_index(name="count")
wards = sorted(theft_by_ward["ward"].dropna().unique())

scatter_sample = (
    df_filtered.dropna(subset=["latitude", "longitude", "ward"])
    .sample(SCATTER_SAMPLE_SIZE, random_state=SEED)
    .copy()
)
scatter_sample["neg_longitude"] = -scatter_sample["longitude"]


# ---------------------------------------------------------------------------
# 3. Build 7 independent figures
# ---------------------------------------------------------------------------
def base_layout(title, height, **kwargs):
    return dict(
        title=title,
        height=height,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=40, t=50, b=60),
        **kwargs,
    )


# Plot 1 — daily
fig1 = go.Figure(
    data=[go.Scatter(x=daily["date"], y=daily["count"], mode="lines",
                     line=dict(color="steelblue", width=1))],
    layout=base_layout("Plot 1 — Daily crime count", 400),
)

# Plot 2 — weekly
fig2 = go.Figure(
    data=[go.Scatter(x=weekly["date"], y=weekly["count"], mode="lines",
                     line=dict(color="darkorange", width=1.5))],
    layout=base_layout("Plot 2 — Weekly crime count", 400),
)

# Plot 3 — monthly
fig3 = go.Figure(
    data=[go.Scatter(x=monthly["date"], y=monthly["count"], mode="lines+markers",
                     line=dict(color="green", width=2))],
    layout=base_layout("Plot 3 — Monthly crime count", 400),
)

# Plot 4 — primary_type counts, log y
fig4 = go.Figure(
    data=[go.Bar(x=type_counts.index, y=type_counts.values, marker_color="mediumpurple")],
    layout=base_layout("Plot 4 — Count by primary_type (log y)", 550),
)
fig4.update_yaxes(type="log", title_text="count (log)")
fig4.update_xaxes(tickangle=45)

# Plot 5 — ward counts
fig5 = go.Figure(
    data=[go.Bar(x=ward_counts.index.astype(str), y=ward_counts.values, marker_color="coral")],
    layout=base_layout("Plot 5 — Count by ward", 500),
)
fig5.update_xaxes(
    title_text="ward", tickangle=90,
    categoryorder="array", categoryarray=[str(w) for w in ward_counts.index],
)
fig5.update_yaxes(title_text="count")

# Plot 6 — monthly THEFT by ward, log y, no legend → static PNG
THEFT_PNG = IMG_DIR / "theft_per_ward.png"
fig6_mpl, ax6 = plt.subplots(figsize=(10, 5))
cmap6 = plt.get_cmap("hsv")
for i, ward in enumerate(wards):
    wd = theft_by_ward[theft_by_ward["ward"] == ward]
    ax6.plot(wd["month"], wd["count"],
             color=cmap6(i / max(len(wards) - 1, 1)), linewidth=1, alpha=0.6)
ax6.set_yscale("log")
ax6.set_ylim(10, 1100)
ax6.set_title("Plot 6 — Monthly THEFT count by ward (log y)")
ax6.set_xlabel("month")
ax6.set_ylabel("count (log)")
ax6.grid(True, which="both", alpha=0.3)
fig6_mpl.tight_layout()
fig6_mpl.savefig(THEFT_PNG, dpi=STATIC_DPI)
plt.close(fig6_mpl)
print(f"Saved {THEFT_PNG}")

# Plot 7 — latitude vs. -longitude, colored by ward, geographic aspect → static PNG
GEO_PNG = IMG_DIR / "geo_scatter.png"
fig7_mpl, ax7 = plt.subplots(figsize=(9, 10))
ax7.scatter(
    scatter_sample["neg_longitude"], scatter_sample["latitude"],
    c=scatter_sample["ward"].astype(int), cmap="tab20",
    s=2, alpha=0.6, linewidths=0,
)
ax7.set_xlim(87.85, 87.5)
ax7.set_ylim(41.65, 42.05)
ax7.set_aspect(1 / cos(radians(41.85)))
ax7.set_xlabel("-longitude")
ax7.set_ylabel("latitude")
ax7.set_title(
    f"Plot 7 — {SCATTER_SAMPLE_SIZE:,} random locations "
    "(latitude vs. -longitude), colored by ward"
)
fig7_mpl.tight_layout()
fig7_mpl.savefig(GEO_PNG, dpi=STATIC_DPI)
plt.close(fig7_mpl)
print(f"Saved {GEO_PNG}")


# ---------------------------------------------------------------------------
# 4. Stack figures into one HTML page with exactly GAP_PX between them
# ---------------------------------------------------------------------------
# Each panel is either a plotly Figure (rendered inline) or a string that is
# already-formed HTML (used for the two static PNG panels).
panels = [
    fig1, fig2, fig3, fig4, fig5,
    f'<img src="img/{THEFT_PNG.name}" alt="Plot 6 — Monthly THEFT count by ward" '
    'style="width:100%;height:auto;display:block;">',
    f'<img src="img/{GEO_PNG.name}" alt="Plot 7 — random locations by ward" '
    'style="width:100%;height:auto;display:block;">',
]

# CSS controls the between-plot gap. The last plot has zero bottom margin
# so that no trailing whitespace exceeds GAP_PX anywhere in the document.
css = f"""
  body {{ font-family: sans-serif; max-width: 1040px; margin: {GAP_PX}px auto; padding: 0 20px; }}
  h1    {{ margin: 0 0 {GAP_PX}px 0; }}
  .plot {{ margin: 0 0 {GAP_PX}px 0; padding: 0; }}
  .plot:last-child {{ margin-bottom: 0; }}
"""

body_parts = ["<h1>Chicago Crime Data — Exploration Dashboard</h1>"]
plotly_count = 0
for panel in panels:
    body_parts.append('<div class="plot">')
    if isinstance(panel, str):
        body_parts.append(panel)
    else:
        include_js = "cdn" if plotly_count == 0 else False
        body_parts.append(panel.to_html(full_html=False, include_plotlyjs=include_js))
        plotly_count += 1
    body_parts.append("</div>")

html = (
    "<!DOCTYPE html>\n"
    "<html lang=\"en\">\n"
    "<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<title>Chicago Crime Data — Exploration Dashboard</title>\n"
    f"<style>{css}</style>\n"
    "</head>\n"
    "<body>\n"
    + "\n".join(body_parts)
    + "\n</body>\n</html>\n"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(html)
print(f"\nSaved dashboard to {OUTPUT_PATH}")
