# validate_model — prompts

**Author:** Joe Hahn <br>
**Email:** jmh.datasciences@gmail.com <br>
**Date:** 2026-April-21 <br>
**branch** main

These prompts are used to generate `validate_model.py`, which loads the trained skforecast model 
produced by `forecast_model.md`, runs multi-horizon backtesting on the validate/forecast data splits, 
and renders an HTML dashboard of model-validation tables and plots.


## Rendering

The dashboard must load quickly on GitHub Pages. Follow these rules:

- All plotly figures share one plotly.js runtime loaded from CDN — do NOT embed plotly.js in the HTML. 
Concretely, pass `include_plotlyjs='cdn'` on the first plotly figure written and 
`include_plotlyjs=False` on every subsequent one.
- For any scatter plot with more than 1,000 points, use `plotly.graph_objects.Scattergl` (WebGL) instead of `Scatter`.

## Load data and model

Load `data/crimes_monthly.csv` into `df_monthly`.

Set `df_train` to all records in `df_monthly` having `TTV = 'train', 
keeping only these columns: `date, year, month, ward, primary_type, delta_count, count_0, count_1, count_2, count_3, count_4`.

Similarly:

- `df_validate` — records with `TTV = 'validate'`, same columns.
- `df_forecast` — records with `TTV = 'forecast'`, same columns.

Report how many records are in `df_train`, `df_validate`, and `df_forecast`.

Load the trained forecaster from `models/forecaster.joblib` into `forecaster`.

## Prediction

Generate forecasts for 1, 2, 3, and 4 months ahead at every validate/forecast date, per `(ward, primary_type)` series, 
using skforecast's backtesting or equivalent rolling-prediction utility:

- For each validate/forecast date `t`, refit-or-reuse the forecaster (refit is optional — document which you chose) 
and predict `count_0` at `t+1, t+2, t+3, t+4`.
- Attach those predictions back onto `df_validate` and `df_forecast` 
as new columns `count_1_pred`, `count_2_pred`, `count_3_pred`, `count_4_pred`, 
so that for every validate/forecast row, the `count_N_pred` column holds the 
forecaster's prediction of what `count_0` will be `N` months after that row's `date`.

Show all records in `df_validate` having `primary_type = THEFT` and `ward = 27`.
Show all records in `df_forecast` having `primary_type = ARSON` and `ward = 27`.

## Dashboard

Create an HTML dashboard of model-validation tables and plots, in this order:

### Plot 1 — total-count timeseries, color-coded by TTV
Start with `df_monthly`, group by `date, TTV`, and compute `sum(count_0)` as `total_count`. Plot:

- the `train` timeseries vs. `date`,
- the `validate` timeseries vs. `date`,
- the `forecast` timeseries vs. `date`.

Use connected scatter plots, color-coded by `TTV`.

### Table 1 — validation scores
Using `df_validate`, compute MAE, RMSE, and R² for each of the four prediction columns (`count_1_pred` vs. `count_1`, 
`count_2_pred` vs. `count_2`, `count_3_pred` vs. `count_3`, `count_4_pred` vs. `count_4`). Render the results as a table.

### Table 2 — feature importances
Extract feature importances from the trained `forecaster` (via `forecaster.get_feature_importances()`, which returns a DataFrame with a `feature` column and an `importance` column). The features include the six autoregressive lags (`lag_1..lag_6`), the two exogenous features (`year`, `month`), and the series encoding. Convert the DataFrame to strings and 
truncate every value in the `importance` column to its first 5 characters. 
Render as a table, ordered by descending importance.

### Plot 2 — count_1_pred vs. count_1 scatterplot
Using `df_validate`, scatter-plot `count_1_pred` (predictions) vs. `count_1` (actuals).

- Logarithmic x-axis from 0.8 to 600.
- Logarithmic y-axis from 0.2 to 600.
- Do not distinguish between different `primary_type` or `ward`.
- Do not add 0.5 to either predictions or actuals.
- Include only points whose prediction AND actual are `> 0`.
- Use **blue** dots with alpha=0.4
- Overplot `y = x` as a dashed line labeled `prediction=actual`.
- Place the legend in the **lower-right corner** of the plot.

### Plot 3 — same as Plot 2 but for `count_2_pred` vs. `count_2`
### Plot 4 — same as Plot 2 but for `count_3_pred` vs. `count_3`
### Plot 5 — same as Plot 2 but for `count_4_pred` vs. `count_4`

### Plot 6 — THEFT timeseries with multi-horizon forecasts
Using `df_validate`, show a timeseries of summed `count_0` for all `primary_type = THEFT` records vs. `date`. 
Color the `count_0` curve **blue** and put vertical error bars on it extending up/down by `sqrt(count_0)`. 
Then overplot:

- summed `count_1_pred` vs. `date + 1 month`
- summed `count_2_pred` vs. `date + 2 months`
- summed `count_3_pred` vs. `date + 3 months`
- summed `count_4_pred` vs. `date + 4 months`

Add a legend in the upper-right corner of this plot.

### Plot 7 — same as Plot 6 but for `primary_type = BURGLARY`

### Plot 8 — same as Plot 6 but for `primary_type = ARSON`

### Plot 9 — per-ward timeseries with forecasts (wards 27, 29, 38)
Using `df_validate`, show a timeseries of summed `count_0` 
across all `primary_type` for wards `27`, `29`, and `38` vs. `date`. 
Put vertical error bars of `sqrt(count_0)` on each `count_0` curve, 
then overplot:

- summed `count_1_pred` vs. `date + 1 month`
- summed `count_2_pred` vs. `date + 2 months`
- summed `count_3_pred` vs. `date + 3 months`
- summed `count_4_pred` vs. `date + 4 months`

Color coding:

- Ward 27 → red
- Ward 29 → blue
- Ward 38 → green

Use a logarithmic y-axis. Add a legend in the upper-right corner. Render as connected scatter plots.

### Plot 10 — THEFT heatmap
Read `data/crimes.csv` and select all records with `primary_type = THEFT` that occurred in 
the most recent complete calendar month in the file. 
Superimpose a heatmap of those thefts on top of a streetmap of Chicago, with:

- **x-axis:** `-longitude`, running from `87.85` (left) to `87.5` (right).
- **y-axis:** `latitude`, running from `41.65` to `42.05`.
- Linear color scaling applied to the binned counts.
- A geographic aspect ratio (so that one degree of latitude and one degree of longitude represent 
equal distances on the ground at Chicago's latitude, ≈ 41.85°).

## Dashboard layout

Stack every plot and table vertically. 
Use **exactly 10 px of vertical margin/padding** (no more, no less) between every pair of adjacent plots, tables, and charts — including between two tables and between a table and a plot. Set gap, margin, and padding to minimal values throughout.

Create a **distinct, self-contained legend** for each individual plot — do not 
share or consolidate legends across plots. 
Every plot must have its own legend embedded within it.

Save the dashboard as `docs/forecast_dashboard.html` (published via GitHub Pages).
