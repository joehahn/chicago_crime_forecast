# Chicago Crime Forecast

Forecasting monthly Chicago crime counts by type and district using seasonal XGBoost and a Keras neural net, with the City of Chicago's public crime dataset.

**Author:** Joe Hahn (jmh.datasciences@gmail.com)

## Setup

```
pip install -r requirements.txt
```

## Workflow

Run the scripts in order:

| Step | Script | Output |
| --- | --- | --- |
| 1. Download data | `python3 get_data.py` | `data/crimes.csv` |
| 2. Explore & visualize | `python3 explore_data.py` | `data_exploration.html` |
| 3. Prep features | `python3 prep_data.py` | `data/crimes_monthly.csv` |
| 4. Train seasonal model | `python3 seasonal_model.py` | `models/seasonal_*.json`, `seasonal_model_dashboard.html` |
| 5. Train neural net | `python3 run_nnet.py` | `models/nnet.keras`, `nnet_dashboard.html` |

Open the generated `*_dashboard.html` files in a browser to inspect model performance.

## Notes

This project was developed with [Claude Code](https://claude.com/claude-code). See `CLAUDE.md` for the per-step prompts used to generate each script.
