# forecast_model — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-17

This is the prompt used to generate `forecast_model.py`, which uses the [skforecast](https://skforecast.org) library to train a classical recursive time-series forecaster on the monthly Chicago crimes data and save the trained forecaster.

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

Save `df_train` to `data/crimes_train.csv`.
Save `df_validate` to `data/crimes_validate.csv`.
Save `df_forecast` to `data/crimes_forecast.csv`.

Show 5 random records from `df_train`.

## Training

Train an skforecast **multi-series recursive forecaster** on the panel of monthly timeseries — one timeseries per `(ward, primary_type)` combination, using `count_0` as the target.

- Use `skforecast.recursive.ForecasterRecursiveMultiSeries` (or the equivalent multi-series class in the installed skforecast version).
- **Underlying regressor:** an XGBoost regressor (`xgboost.XGBRegressor`) with `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `n_jobs=-1`.
- **Autoregressive lags:** 6 (i.e., the last 6 months of each series).
- **Exogenous features:** `year`, `month` (both per-date, shared across all series).
- **Historical window used to fit the forecaster:** all records whose `date < 2025-01-01` (i.e., every record where `TTV = 'train'`). skforecast needs contiguous per-series history, so the date-based cut is the correct one.
- Name the trained forecaster `forecaster`.
- Save `forecaster` to `models/forecaster.joblib` (or the appropriate file extension for skforecast's save helper).
