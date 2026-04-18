# forecast_model — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-17

This is the prompt used to generate `forecast_model.py`, which uses the [skforecast](https://skforecast.org) library to train a classical recursive time-series forecaster on the monthly Chicago crimes data and produces an HTML dashboard of model-validation tables and plots. Results are intended to be directly comparable to those from `seasonal_model.py` and `run_nnet.py`.

## Ground rules

Execute the following completely from scratch, starting fresh:

- Do not use any cached results, temporary files, or previously computed outputs.
- Do not gather any files using git.
- Do not use any files previously stored in `/tmp`.
- Clear any cache files first, then execute everything fresh.

## Prompt

Load `data/crimes_monthly.csv` into `df_monthly`.

Set `df_train` to all records in `df_monthly` having `TTV = 'train'`, keeping only these columns: `date, year, month, ward, primary_type, delta_count, count_0, count_1, count_2, count_3, count_4`.

Similarly:

- `df_test` — records with `TTV = 'test'`, same columns.
- `df_validate` — records with `TTV = 'validate'`, same columns.
- `df_forecast` — records with `TTV = 'forecast'`, same columns.

Report how many records are in `df_train`, `df_test`, `df_validate`, and `df_forecast`.

Show 5 random records from `df_train`.

## Training

Train an skforecast **multi-series recursive forecaster** on the panel of monthly timeseries — one timeseries per `(ward, primary_type)` combination, using `count_0` as the target.

- Use `skforecast.recursive.ForecasterRecursiveMultiSeries` (or the equivalent multi-series class in the installed skforecast version).
- **Underlying regressor:** an XGBoost regressor (`xgboost.XGBRegressor`). Use hyperparameters similar to those in `seasonal_model.py` (`n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `n_jobs=-1`) so results are comparable.
- **Autoregressive lags:** 6 (i.e., the last 6 months of each series).
- **Exogenous features:** `year`, `month` (both per-date, shared across all series).
- **Historical window used to fit the forecaster:** all records whose `date < 2025-01-01` (i.e., every record where `TTV ∈ {'train', 'test'}`). Note: skforecast needs contiguous per-series history, so splitting on the random `TTV` label alone would leave holes — the date-based cut is the correct one.
- Name the trained forecaster `forecaster`.
- Save `forecaster` to `models/forecaster.joblib` (or the appropriate file extension for skforecast's save helper).

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

- the summed train + test timeseries vs. `date`,
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
