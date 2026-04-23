# Chicago Crime Forecast

Forecasting monthly Chicago crime counts by type and ward using an skforecast recursive multi-series forecaster (XGBoost regressor under the hood), with the City of Chicago's public crime dataset.

**Author:** Joe Hahn (jmh.datasciences@gmail.com)

## Setup

```
pip3 install -r requirements.txt
```

## Workflow

Run the scripts in order:

| Step | Script | Output |
| --- | --- | --- |
| 1. Download data | `python3 get_data.py` | `data/crimes.csv` |
| 2. Explore & visualize | `python3 explore_data.py` | [`docs/data_exploration.html`](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html) |
| 3. Prep features | `python3 prep_data.py` | `data/crimes_monthly.csv` |
| 4. Train skforecast model | `python3 forecast_model.py` | `models/forecaster.joblib`, `data/crimes_{train,validate,forecast}.csv` |
| 5. Validate model | `python3 validate_model.py` | [`docs/forecast_dashboard.html`](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html), `forecast_validate.ipynb` (notebook that rebuilds & validates the forecaster end-to-end) |

## Dashboards

Interactive dashboards are published via GitHub Pages:

- [Data exploration](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html)
- [skforecast validation](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html)

## Notes

This project was developed with [Claude Code](https://claude.com/claude-code). See `CLAUDE.md` for the per-step prompts used to generate each script.

Earlier seasonal-XGBoost and Keras neural-net experiments are archived under `old/`.
