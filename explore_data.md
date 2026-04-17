# explore_data — prompt

**Author:** Joe Hahn (jmh.datasciences@gmail.com)
**Date:** 2026-April-8

This is the prompt used to generate `explore_data.py`, which reads the Chicago crimes dataset, profiles and cleans the data, and produces an HTML dashboard of exploratory plots.

## Prompt

Read `data/crimes.csv` into `df_filtered`. Report how many records are in `df_filtered`.

Profile `df_filtered`.

Create a data-exploration dashboard showing the following plots of data in `df_filtered`:

- **Plot 1** — timeseries of daily record count vs. date.
- **Plot 2** — timeseries of weekly record count vs. date.
- **Plot 3** — timeseries of monthly record count vs. date.
- **Plot 4** — bar chart of `primary_type` counts (vertical bars, descending order, logarithmic y-axis).
- **Plot 5** — bar chart of `ward` counts (vertical bars, descending order).
- **Plot 6** — timeseries of count of `primary_type = THEFT` vs. time, one trace per `ward`. Logarithmic y-axis running from 10 to 1100. Do not show a legend.
- **Plot 7** — scatter plot of 10,000 random records: latitude vs. `-longitude`, colorized by `ward`. x-axis from 87.85 (left) to 87.5 (right). y-axis from 41.65 to 42.05. Apply a geographic aspect ratio.

Stack each plot vertically. Spread the plots out so that one plot does not run into the legend of the plot above it.

Save the dashboard as `docs/data_exploration.html` (published via GitHub Pages).
