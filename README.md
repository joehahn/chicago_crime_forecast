# Chicago Crime Forecast

**One model that forecasts a thousand time series at once.** Chicago publishes
every reported crime as open data; this pipeline turns that feed into
month-ahead counts for all 1,000 ward-by-crime-type pairs in the city — 50
wards × 20 offense types — using a single skforecast recursive multi-series
forecaster with an XGBoost regressor underneath.

The domain here is municipal crime data, but the pattern is general: when you
have many related series that are each too short and too noisy to model on
their own, you pool them into one forecaster that learns the shared seasonality
and lets the series identity carry the rest. The accuracy claim is a
rolling-origin backtest at a four-month horizon — error measured on months the
model never saw, not on its own training data.

Built end-to-end with [Claude Code](https://claude.com/claude-code) by
**Joseph M. Hahn, Ph.D.**, an independent AI and machine learning consultant —
[jmh-datasciences.com](https://jmh-datasciences.com) ·
[LinkedIn](https://www.linkedin.com/in/hahnjoe/) · joe.hahn@jmh-datasciences.com

> **A research demonstration, not a policing tool.** This forecasts aggregate
> monthly counts per ward from the City of Chicago's published open data. It is
> not a risk-scoring, predictive-policing, or resource-allocation system, it
> says nothing about individuals or addresses, and it should not be used to
> direct enforcement.

**License:** [MIT](LICENSE) — code and writing both, do what you like with them.  
**Started:** 2026-Apr-07 · **Branch:** `main`

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
| 4. Train skforecast model | `python3 forecast_model.py` | `models/forecaster.joblib` |
| 5. Validate model | `python3 validate_model.py` | [`docs/forecast_dashboard.html`](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html), `forecast_validate.ipynb` (notebook that rebuilds & validates the forecaster end-to-end) |

## Dashboards

Interactive dashboards are published via GitHub Pages:

- [Data exploration](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html)
- [skforecast validation](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html)

## Notes

This project was developed with [Claude Code](https://claude.com/claude-code). See `CLAUDE.md` for the per-step prompts used to generate each script.

Earlier seasonal-XGBoost and Keras neural-net experiments are archived under `old/`.

## About the author

I am Joseph M. Hahn, Ph.D., an independent AI and machine learning consultant. Through **JMH DataSciences** I build production AI and machine learning systems for clients who need a real decision automated, not a demo. Before going independent I spent eight years inside Oracle's AI Center of Excellence delivering AI systems for enterprise clients in manufacturing, oil and gas, public sector, and retail, and before that four years building machine learning systems on large data platforms.

This repo is one of several demonstrations of the same underlying pattern: **take a messy public data feed, automate a judgment someone would otherwise make by hand, and then build the scaffolding that proves the result isn't fooling itself.** Here the judgment is a forecast and the scaffolding is the rolling-origin backtest — because the easy version of this project is a model that scores beautifully on data it has already seen, and the difference between that and a number you can act on is most of the work. If that shape matches a problem in your business, the work I do and what it costs are at [jmh-datasciences.com](https://jmh-datasciences.com).

**Related work:**

- [**diplomacy-A2A**](https://github.com/joehahn/diplomacy-A2A) — seven Claude-powered agents play *Diplomacy*, the classic seven-player negotiation board game, against each other: forming alliances, bargaining, and betraying each other over the A2A protocol.
- [**geo-herd-rider**](https://github.com/joehahn/geo-herd-rider) — an AI agent reads a continuous feed of unstructured news and makes a routine judgment call a person would otherwise make by hand, while a deterministic optimizer handles everything that has to be auditable.
- [**portfolio-wave-rider**](https://github.com/joehahn/portfolio-wave-rider) — a mechanical retriever gathers the articles, an LLM judges which of them matter, and a mean-variance optimizer sizes the result: judgment confined to the middle stage.

## License

[MIT](LICENSE) — the code, the notebooks, the writing, and the published dashboards. Use them for anything, commercially or not, with or without attribution. Credit is appreciated but not required; the form I'd suggest is *Joseph M. Hahn, Ph.D., JMH DataSciences — https://jmh-datasciences.com*.

**Source data.** The underlying incident records come from the [City of Chicago's public crime dataset](https://www.chicago.gov/city/en/dataset/crime.html) via the Socrata open-data API. That data belongs to the City and is governed by its terms of use, not by the license above. Raw and derived CSVs are gitignored and regenerated by `get_data.py`; the dashboards under `docs/` display counts aggregated from it.
