# get_data — prompts

**Author:** Joe Hahn
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-8
**branch** main

These prompts generate `get_data.py`, which downloads the Chicago crimes dataset 
and performs some very light filtering of that data to make it business friendly. 
See the [Chicago Crime dataset page](https://www.chicago.gov/city/en/dataset/crime.html) 
for a description of the source data.

## Prompts

Download the latest crime data from the Chicago Data Portal at `data.cityofchicago.org` 
using the Socrata API endpoint for the Crimes dataset (dataset ID: `ijzp-q8t2`). 
Fetch records from the past 4 years and store the result in object `df_raw`. 
Always re-download from the source; do not use cached data stored locally. 
State how many columns and records are in `df_raw`.

Inspect each text column in `df_raw` (such as the `description` column) and mitigate any problematic carriage returns. 
Store the result in `df_clean`. State how many records are in `df_clean`.

Display a count of `primary_type` in `df_clean`.

Filter out from `df_clean` any records whose `primary_type` is one of:

- `CRIMINAL SEXUAL ASSAULT`
- `OFFENSE INVOLVING CHILDREN`
- `SEX OFFENSE`
- `PROSTITUTION`

Store the result in `df_filtered`. Report how many records are in `df_filtered`.

Display 1 random record from `df_filtered`.

Report the minimum and maximum values of `date`.

Save `df_filtered` to `data/crimes.csv`.
