# CLAUDE.md
**Author:** Joe Hahn  
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-7
**branch** main

## Project

This repository is for the Chicago crime-forcasting project.
This project's github is at https://github.com/joehahn/chicago_crime_forecast
Always use python3 instead of python when running scripts


## 1. download and inspect the data

Begin by prompting claude to download chicago crime data

    @get_data.txt

Then prompt claude to explore and visualize crime data

    @explore_data.txt

and inspect the plots that the above published in data_exploration.html


## 2. prepare the data for training ML models


Tell claude to create the prep_data script

    @prep_data.txt


## 3. train the so-called seasonal ML model


Train the seasonal ML model on the output of the above via

    @seasonal_model.txt

and inspect seasonal_model_dashboard.html to see various model validation reports


## 4. train a simple neural network using Keras/Tensorflow

Train the neural network model

    @seasonal_model.txt

and inspect nnet_dashboard.html to see model validation reports


## 5. train a traditional forecasting ML model

    @forecast_model.txt

and inspect forecast_dashboard.html to see model validation reports
