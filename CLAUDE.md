# CLAUDE.md
by Joe Hahn
jmh.datasciences@gmail.com
2026-April-7
branch=main

## Project

This repository is for the Chicago crime-forcasting project.
This project's github is at https://github.com/joehahn/chicago_crime_forecast
Always use python3 instead of python when running scripts

To start Claude:

    claude

Begin by prompting claude to download chicago crime data

    @get_data.txt

Tehn prompt claude to explore and visualize crime data

    @explore_data.txt

and inspect the plots that the above published in data_exploration.html

Then tell claude to create the prep_data script

    @prep_data.txt

Then train the seasonal ML model on the output of the above via

    @seasonal_model.txt

and...



