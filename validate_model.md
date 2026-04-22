# validate_model — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-21

This is the prompt used to generate `validate_model.py`, which loads the trained skforecast model produced by `forecast_model.md`, runs multi-horizon backtesting on the validate/forecast data splits, and renders an HTML dashboard of model-validation tables and plots.

## Ground rules

Execute the following completely from scratch, starting fresh:

- Do not use any cached results, temporary files, or previously computed outputs.
- Do not gather any files using git.
- Do not use any files previously stored in `/tmp`.
- Clear any cache files first, then execute everything fresh.

## Rendering

The dashboard must load quickly on GitHub Pages. Follow these rules:

- All plotly figures share one plotly.js runtime loaded from CDN — do NOT embed plotly.js in the HTML. Concretely, pass `include_plotlyjs='cdn'` on the first plotly figure written and `include_plotlyjs=False` on every subsequent one.
- For any scatter plot with more than 1,000 points, use `plotly.graph_objects.Scattergl` (WebGL) instead of `Scatter`.

## Prompt

Load the three CSVs produced by `forecast_model.md`:

- `data/crimes_train.csv` → `df_train`
- `data/crimes_validate.csv` → `df_validate`
- `data/crimes_forecast.csv` → `df_forecast`

Report how many records are in `df_train`, `df_validate`, and `df_forecast`.

Reconstruct `df_monthly` by concatenating `df_train`, `df_validate`, and `df_forecast` and tagging each with a new `TTV` column set to `'train'`, `'validate'`, or `'forecast'` respectively. Name the combined frame `df_monthly`.

Load the trained forecaster from `models/forecaster.joblib` into `forecaster`.

## Prediction

Generate forecasts for 1, 2, 3, and 4 months ahead at every validate/forecast date, per `(ward, primary_type)` series, using skforecast's backtesting or equivalent rolling-prediction utility:

- For each validate/forecast date `t`, refit-or-reuse the forecaster (refit is optional — document which you chose) and predict `count_0` at `t+1, t+2, t+3, t+4`.
- Attach those predictions back onto `df_validate` and `df_forecast` as new columns `count_1_pred`, `count_2_pred`, `count_3_pred`, `count_4_pred`, so that for every validate/forecast row, the `count_N_pred` column holds the forecaster's prediction of what `count_0` will be `N` months after that row's `date`.
- Do not write any predictions to CSV files.

Show all records in `df_validate` having `primary_type = THEFT` and `ward = 27`.
Show all records in `df_forecast` having `primary_type = THEFT` and `ward = 27`.

## Dashboard

Create an HTML dashboard of model-validation tables and plots, in this order:

### Plot 1 — total-count timeseries, color-coded by TTV
Start with `df_monthly`, group by `date, TTV`, and compute `sum(count_0)` as `total_count`. Plot:

- the `train` timeseries vs. `date`,
- the `validate` timeseries vs. `date`,
- the `forecast` timeseries vs. `date`.

Use connected scatter plots, color-coded by `TTV`.

### Table 1 — validation scores
Using `df_validate`, compute MAE, RMSE, and R² for each of the four prediction columns (`count_1_pred` vs. `count_1`, `count_2_pred` vs. `count_2`, `count_3_pred` vs. `count_3`, `count_4_pred` vs. `count_4`). Render the results as a table.

### Plot 2 — THEFT timeseries with multi-horizon forecasts
Using `df_validate`, show a timeseries of summed `count_0` for all `primary_type = THEFT` records vs. `date`. Color the `count_0` curve **blue** and put vertical error bars on it extending up/down by `sqrt(count_0)`. Then overplot:

- summed `count_1_pred` vs. `date + 1 month`
- summed `count_2_pred` vs. `date + 2 months`
- summed `count_3_pred` vs. `date + 3 months`
- summed `count_4_pred` vs. `date + 4 months`

Add a legend in the upper-right corner of this plot.

### Plot 3 — per-ward timeseries with forecasts (wards 27, 29, 38)
Using `df_validate`, show a timeseries of summed `count_0` across all `primary_type` for wards `27`, `29`, and `38` vs. `date`. Put vertical error bars of `sqrt(count_0)` on each `count_0` curve, then overplot:

- summed `count_1_pred` vs. `date + 1 month`
- summed `count_2_pred` vs. `date + 2 months`
- summed `count_3_pred` vs. `date + 3 months`
- summed `count_4_pred` vs. `date + 4 months`

Color coding:

- Ward 27 → red
- Ward 29 → blue
- Ward 38 → green

Use a logarithmic y-axis. Add a legend in the upper-right corner. Render as connected scatter plots.

## Dashboard layout

Stack every plot and table vertically. Use **exactly 10 px of vertical margin/padding** (no more, no less) between every pair of adjacent plots, tables, and charts — including between two tables and between a table and a plot. Set gap, margin, and padding to minimal values throughout.

Create a **distinct, self-contained legend** for each individual plot — do not share or consolidate legends across plots. Every plot must have its own legend embedded within it.

Save the dashboard as `docs/forecast_dashboard.html` (published via GitHub Pages).
