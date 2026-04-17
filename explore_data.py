#!/usr/bin/env python3
"""Profile the Chicago crimes dataset and render an HTML exploration dashboard.

Reads  : data/crimes.csv                    (produced by get_data.py)
Writes : docs/data_exploration.html         (published via GitHub Pages)

Layout: each plot is rendered as an independent Plotly figure, then stacked
in a single HTML page with exactly 20 px of vertical margin between adjacent
plots (enforced via CSS, not via Plotly's subplot `vertical_spacing`).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SEED = 42
SCATTER_SAMPLE_SIZE = 10_000
GAP_PX = 20  # exactly 20 px between every pair of adjacent plots

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "data" / "crimes.csv"
OUTPUT_PATH = ROOT / "docs" / "data_exploration.html"


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

# Plot 6 — monthly THEFT by ward, log y, no legend
colors = [f"hsl({int(i / len(wards) * 360)},70%,50%)" for i in range(len(wards))]
fig6 = go.Figure(layout=base_layout("Plot 6 — Monthly THEFT count by ward (log y)", 500))
for i, ward in enumerate(wards):
    wd = theft_by_ward[theft_by_ward["ward"] == ward]
    fig6.add_trace(go.Scatter(
        x=wd["month"], y=wd["count"], mode="lines",
        line=dict(color=colors[i], width=1),
    ))
fig6.update_yaxes(type="log", range=[1, 3.041], title_text="count (log)")

# Plot 7 — latitude vs. -longitude, colored by ward, geographic aspect
fig7 = go.Figure(
    data=[go.Scatter(
        x=scatter_sample["neg_longitude"],
        y=scatter_sample["latitude"],
        mode="markers",
        marker=dict(
            color=scatter_sample["ward"],
            colorscale="Turbo",
            size=3,
            opacity=0.6,
            colorbar=dict(title="ward"),
        ),
    )],
    layout=base_layout(
        f"Plot 7 — {SCATTER_SAMPLE_SIZE:,} random locations (latitude vs. -longitude), colored by ward",
        800,
    ),
)
fig7.update_xaxes(range=[87.85, 87.5], title_text="-longitude")
fig7.update_yaxes(
    range=[41.65, 42.05], title_text="latitude",
    scaleanchor="x", scaleratio=1.34,  # 1/cos(41.85°) ≈ geographic aspect
)


# ---------------------------------------------------------------------------
# 4. Stack figures into one HTML page with exactly GAP_PX between them
# ---------------------------------------------------------------------------
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]

# CSS controls the between-plot gap. The last plot has zero bottom margin
# so that no trailing whitespace exceeds GAP_PX anywhere in the document.
css = f"""
  body {{ font-family: sans-serif; max-width: 1040px; margin: {GAP_PX}px auto; padding: 0 20px; }}
  h1    {{ margin: 0 0 {GAP_PX}px 0; }}
  .plot {{ margin: 0 0 {GAP_PX}px 0; padding: 0; }}
  .plot:last-child {{ margin-bottom: 0; }}
"""

body_parts = ["<h1>Chicago Crime Data — Exploration Dashboard</h1>"]
for i, fig in enumerate(figs):
    include_js = "cdn" if i == 0 else False
    body_parts.append('<div class="plot">')
    body_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
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
