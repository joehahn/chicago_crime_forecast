# CLAUDE.md
**Author:** Joe Hahn  
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-7 <br>
**branch** main

## Project

This repository is for the Chicago crime-forecasting project.
This project was developed using Claude Code.
This project's github is at https://github.com/joehahn/chicago_crime_forecast

## Ground rules

When executing any prompt files, execute completely from scratch and starting fresh:

- Do not use any cached results, temporary files, or previously computed outputs.
- Do not pull any files using git.
- Do not use any files previously stored in `/tmp`.
- Ignore any files stored in the `blog` and `old` folders.
- Clear any cache files first, then execute everything fresh.
- When a prompt script says to display something, display it inline, do not skip
- Always write simple code that is well commented and understood at a glance

## 1. download and inspect the data

Begin by prompting claude to download chicago crime data

    @get_data.md

Then prompt claude to explore and visualize crime data

    @explore_data.md

and inspect the plots that the above published in [docs/data_exploration.html](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html)


## 2. prepare the data for training ML models


Tell claude to create the prep_data script

    @prep_data.md


## 3. train an sklearn forecasting ML model

    @forecast_model.md


## 4. validate the forecasting model

    @validate_model.md

and inspect [docs/forecast_dashboard.html](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html) to see model validation reports.

This step also generates `forecast_validate.ipynb`, a Jupyter notebook that replicates `forecast_model.py` + `validate_model.py` end-to-end — retraining the skforecast model, running the rolling-origin backtest, and rebuilding the dashboard. Open it with `jupyter lab` to step through the full forecast+validate pipeline interactively.


## Todo:


## 5. summarize the above via blog post


Earlier seasonal-XGBoost and Keras neural-net experiments are archived under `old/`.

