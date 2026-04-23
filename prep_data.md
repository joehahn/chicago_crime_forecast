# prep_data — prompts

**Author:** Joe Hahn
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-8
**branch** main

These prompts generate `prep_data.py`, which prepares the Chicago crime data for training ML models 
that forecast crime counts across the city.

## Prompts

Load `data/crimes.csv` into `df_filtered`.

Report the shape of `df_filtered`.

Starting from `df_filtered`, rename the `date` column as `timestamp`. 
Derive the month of the year from the `timestamp` column and name the new column `month`. 
Then keep only those records whose `primary_type` are among the top 20. 
Store the result in `df_20`. 

Report how many records are in `df_20`.

Display counts of `primary_type` in `df_20`.

Display 1 random record in `df_20`, showing all columns.

Show the column types in `df_20`.

State the minimum and maximum date in `df_20`.

Group `df_20` by `year, month, ward, primary_type` and compute:

- `mean(arrest)`
- `mean(domestic)`
- `mean(latitude)`
- `mean(longitude)`
- `count(id)` as `count_0`

Drop the `mean_` prefix from the new columns. 
Cast `ward` to integer. 
Add a column `day` containing integer `1`. 
Reorder columns so that `count_0` is last. 
Order records by `year, month, ward, primary_type`. 
Name this result `df_avg`. 
Report how many records are in `df_avg`.

Set `df_date = df_avg` and add a new column `date` derived from the `year`, `month`, and `day` columns.

Which ward in `df_date` has the greatest `sum(count_0)` among `primary_type = THEFT` records?

Show all records in `df_date` having `primary_type = THEFT` and `ward = 42`.

Show all records in `df_date` having `primary_type = ARSON` and `ward = 42`.

Use the `date`, `ward`, `primary_type` columns to determine which records are missing from `df_date`. 
Zero-pad those missing records in `df_date` with `count_0` set to `0`. 
Be sure to also include the missing values in the `year` and `month` columns. 
Order all records by `date, ward, primary_type`. Store the result in `df_pad`. 
Report how many records are in `df_pad`.

Show all records in `df_pad` having `primary_type = ARSON` and `ward = 42`.

For each column in `df_pad`, find any NaN values and replace them with random selections 
of non-NaN values from the same column. Store the result in `df_nan`.

Show all records in `df_nan` having `primary_type = ARSON` and `ward = 22`.

Print 5 random records in `df_nan`.

Partition `df_nan` by `year, month, ward, primary_type` and order by `date`. Append:

- `count_previous` — `count_0` shifted backwards in time by 1 month
- `count_1` — `count_0` shifted forwards by 1 month
- `count_2` — shifted forwards by 2 months
- `count_3` — shifted forwards by 3 months
- `count_4` — shifted forwards by 4 months

Set `delta_count = count_0 - count_previous`. 
Do not drop any records due to missing data; 
use `NaN` to indicate missing future data. 
Reorder columns so `date` is first and 
`delta_count, count_0, count_1, count_2, count_3, count_4` are last. 
Name the result `df_target`.

Print all records in `df_target` having `primary_type = THEFT` and `ward = 27`.

Start with `df_target` and append a column `ran_num` containing random floats uniformly distributed on [0, 1); 
call the result `df_ttv`. Create a new column `TTV` such that:

- rows with `ran_num <= 0.667` → `TTV = 'train'`
- rows with `ran_num  > 0.667` → `TTV = 'test'`

Then set `TTV = 'validate'` for all rows with `date >= '2025-01-01'`.

Set `last_few_dates` to the 2 greatest dates in `df_ttv`, 
and set `TTV = 'forecast'` for records whose `date` is in `last_few_dates`.

Then set `last_few_dates` to the greatest date in `df_ttv`, 
and set `TTV = 'incomplete'` for records whose `date` is in `last_few_dates`.

Drop the following columns from `df_ttv`: `arrest`, `domestic`, `count_previous`, ran_num`. 
Name the result `df_monthly`.

Drop from `df_monthly` all records having `delta_count = NaN`.

Pretty-print ALL records in `df_monthly` having `primary_type = THEFT` and `ward = 22`

Save `df_monthly` as `data/crimes_monthly.csv`.
