# Chicago Crime Forecast

Forecasting monthly Chicago crime counts by type and district using a seasonal XGBoost baseline, a Keras neural net, and an skforecast recursive multi-series forecaster, with the City of Chicago's public crime dataset.

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
| 4. Train seasonal model | `python3 seasonal_model.py` | `models/seasonal_*.json`, [`docs/seasonal_model_dashboard.html`](https://joehahn.github.io/chicago_crime_forecast/seasonal_model_dashboard.html) |
| 5. Train neural net | `python3 run_nnet.py` | `models/nnet.keras`, [`docs/nnet_dashboard.html`](https://joehahn.github.io/chicago_crime_forecast/nnet_dashboard.html) |
| 6. Train skforecast model | `python3 forecast_model.py` | `models/forecaster.joblib`, [`docs/forecast_dashboard.html`](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html) |

## Dashboards

Interactive dashboards are published via GitHub Pages:

- [Data exploration](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html)
- [Seasonal model validation](https://joehahn.github.io/chicago_crime_forecast/seasonal_model_dashboard.html)
- [Neural net validation](https://joehahn.github.io/chicago_crime_forecast/nnet_dashboard.html)
- [skforecast validation](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html)

## Notes

This project was developed with [Claude Code](https://claude.com/claude-code). See `CLAUDE.md` for the per-step prompts used to generate each script.
