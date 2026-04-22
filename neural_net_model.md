# neural_net_model — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-15

This is the prompt used to generate `run_nnet.py`, which uses TensorFlow/Keras to train a simple multi-output neural network that forecasts Chicago crime counts, and produces an HTML dashboard of model-validation tables and plots.

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

- `df_validate` — records with `TTV = 'validate'`, same columns.
- `df_forecast` — records with `TTV = 'forecast'`, same columns.

Report how many records are in `df_train`, `df_validate`, and `df_forecast`.

Show 5 random records from `df_train`.

## Training

Train a simple Keras/TensorFlow multilayer perceptron (MLP) with **one hidden layer**.

- **Inputs (features):** `year, month, ward, primary_type, delta_count, count_0` from `df_train`.
- **Outputs (targets):** `count_1, count_2, count_3, count_4` — a single multi-output regression.
- **Hidden layer size:** choose whatever is most appropriate for this number of inputs and outputs; use your best judgment.
- **Validation set:** use `df_validate` as the evaluation set during training.
- Name the trained model `nnet`.
- Save `nnet` under `models/`.

## Prediction

For each of `df_validate` and `df_forecast`:

- Use `nnet` to predict, and append the predictions as new columns `count_1_pred`, `count_2_pred`, `count_3_pred`, `count_4_pred`.
- Do not write any predictions to CSV files.

Show 5 random records from `df_validate`.

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

### Table 2 — feature importances
Compute permutation feature importances for `nnet` on `df_validate` using `sklearn.inspection.permutation_importance`, separately for each output target (`count_1`, `count_2`, `count_3`, `count_4`). Put the results into a DataFrame with one row per feature and one column per target. Convert the DataFrame to strings and then truncate every value to its first 5 characters. Render as a table.

### Plot 4 — nnet predictions vs. actuals, target 1
Using `df_validate`, scatter-plot `count_1_pred` (predictions) vs. `count_1` (actuals).

- Logarithmic x-axis from 0.8 to 600.
- Logarithmic y-axis from 0.2 to 600.
- Do not distinguish between different `primary_type` or `ward`.
- Do not add 0.5 to either predictions or actuals.
- Include only points whose prediction AND actual are `> 0`.
- Use opaque **green** dots for records whose `count_0` falls in the middlemost 80% of the data, and opaque **red** dots for records in the outermost 20%.
- Overplot `y = x` as a dashed line labeled `prediction=actual`.
- Place the legend in the **lower-right corner** of the plot.

### Plot 5 — same as Plot 4 but for target 2 (`count_2_pred` vs. `count_2`)
### Plot 6 — same as Plot 4 but for target 3 (`count_3_pred` vs. `count_3`)
### Plot 7 — same as Plot 4 but for target 4 (`count_4_pred` vs. `count_4`)

### Plot 8 — same as Plot 2 but for `primary_type = BURGLARY`
### Plot 9 — same as Plot 2 but for `primary_type = ARSON`

### Plot 10 — THEFT heatmap
Read `data/crimes.csv` and select all records with `primary_type = THEFT` that occurred in March 2026. Superimpose a heatmap of those THEFTS on top of a streetmap of Chicago, with:

- **x-axis:** `-longitude`, running from `87.85` (left) to `87.5` (right).
- **y-axis:** `latitude`, running from `41.65` to `42.05`.
- Logarithmic color scaling applied to the binned counts.
- A geographic aspect ratio (so that one degree of latitude and one degree of longitude represent equal distances on the ground at Chicago's latitude, ≈ 41.85°).

## Dashboard layout

Stack every plot and table vertically. Use **exactly 10 px of vertical margin/padding** (no more, no less) between every pair of adjacent plots, tables, and charts — including between two tables and between a table and a plot. Set gap, margin, and padding to minimal values throughout.

Create a **distinct, self-contained legend** for each individual plot — do not share or consolidate legends across plots. Every plot must have its own legend embedded within it.

Save the dashboard as `docs/nnet_dashboard.html` (published via GitHub Pages).
