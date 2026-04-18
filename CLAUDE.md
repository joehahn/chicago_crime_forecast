# CLAUDE.md
**Author:** Joe Hahn  
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-7 <br>
**branch** main

## Project

This repository is for the Chicago crime-forcasting project.
This project's github is at https://github.com/joehahn/chicago_crime_forecast
Always use python3 instead of python when running scripts.
Ignore any files in the blog folder.
This project was developed using Claude Code


## 1. download and inspect the data

Begin by prompting claude to download chicago crime data

    @get_data.md

Then prompt claude to explore and visualize crime data

    @explore_data.md

and inspect the plots that the above published in [docs/data_exploration.html](https://joehahn.github.io/chicago_crime_forecast/data_exploration.html)


## 2. prepare the data for training ML models


Tell claude to create the prep_data script

    @prep_data.md


## 3. train the so-called seasonal ML model


Instruct Claude to train the seasonal ML model on the output of the above via

    @seasonal_model.md

and inspect [docs/seasonal_model_dashboard.html](https://joehahn.github.io/chicago_crime_forecast/seasonal_model_dashboard.html) to see various model validation reports


## 4. train a simple neural network using Keras/Tensorflow

Prompt Claude to train the neural network model

    @neural_net_model.md

and inspect [docs/nnet_dashboard.html](https://joehahn.github.io/chicago_crime_forecast/nnet_dashboard.html) to see model validation reports


## Todo:


## 5. train a traditional forecasting ML model

    @forecast_model.md

and inspect [docs/forecast_dashboard.html](https://joehahn.github.io/chicago_crime_forecast/forecast_dashboard.html) to see model validation reports

## 6. summarize the above via blog post

