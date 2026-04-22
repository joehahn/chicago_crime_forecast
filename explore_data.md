# explore_data — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-8

This is the prompt used to generate `explore_data.py`, which reads the Chicago crimes dataset, profiles and cleans the data, and produces an HTML dashboard of exploratory plots.

## Rendering

The dashboard must load quickly on GitHub Pages. Follow these rules:

- All interactive plotly figures share one plotly.js runtime loaded from CDN — do NOT embed plotly.js in the HTML. Concretely, pass `include_plotlyjs='cdn'` on the first plotly figure written and `include_plotlyjs=False` on every subsequent one.
- For any scatter plot with more than 1,000 points, use `plotly.graph_objects.Scattergl` (WebGL) instead of `Scatter`.
- For plots flagged as **static PNG** below, render them with `matplotlib`, save as PNG under `docs/img/` at `dpi=120` or higher (legible at ~1000 px wide), and embed in the dashboard as an `<img>` tag. Do NOT use plotly for these.

## Prompt

Read `data/crimes.csv` into `df_filtered`. Report how many records are in `df_filtered`.

Profile `df_filtered`.

Create a data-exploration dashboard showing the following plots of data in `df_filtered`:

- **Plot 1** (interactive plotly) — timeseries of daily record count vs. date. If the daily series has more than 2,000 points, aggregate to weekly first.
- **Plot 2** (interactive plotly) — timeseries of weekly record count vs. date.
- **Plot 3** (interactive plotly) — timeseries of monthly record count vs. date.
- **Plot 4** (interactive plotly) — bar chart of `primary_type` counts (vertical bars, descending order, logarithmic y-axis).
- **Plot 5** (interactive plotly) — bar chart of `ward` counts (vertical bars, descending order).
- **Plot 6** (**static PNG** → `docs/img/theft_per_ward.png`) — timeseries of count of `primary_type = THEFT` vs. time, one line per `ward`, using `matplotlib`. Logarithmic y-axis running from 10 to 1100. Use muted colors (e.g. `alpha=0.6`) and no legend.
- **Plot 7** (**static PNG** → `docs/img/geo_scatter.png`) — scatter plot of 10,000 random records using `matplotlib`: latitude vs. `-longitude`, colorized by `ward` with a discrete colormap (e.g. `tab20`). x-axis from 87.85 (left) to 87.5 (right). y-axis from 41.65 to 42.05. Set the axes aspect ratio to `1/cos(radians(41.85))` so one degree of latitude and one degree of longitude represent equal distances on the ground at Chicago's latitude. Use slightly larger, very opaque markers. (Ward-hover tooltips are intentionally dropped as a tradeoff for the speedup.)

Stack each plot vertically with **exactly 10 px of vertical margin/padding (no more, no less) between every pair of adjacent plots, tables, and charts**, so that no element (including axis labels, tick labels, or legends) ever touches or overlaps the one above or below it.

Save the dashboard as `docs/data_exploration.html` (published via GitHub Pages).
