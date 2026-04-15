# neural_net_model.md
**Author:** Joe Hahn  
**Email:** jmh.datasciences@gmail.com  
**Date:** 2026-April-15

The following will use Tensorflow to train a simple neural network for forecast crimes across Chicago

---

Execute the following completely from scratch and starting fresh.
Do not use any cached results, temporary files, or previously computed outputs. 
Do not gather any files using git.
Do not utlize any files previously stored in /tmp
Clear any cache files first, then execute everything fresh.

---

## 1. read the crime data from file

Begin by loading file data/crimes_monthly.csv into object df_monthly

And then set df_train = all records in df_monthly having TTV='train', 
keeping only the date, year, month, ward, primary_type, delta_count, count_0, count_1, count_2, count_3, count_4 columns.

Set df_test = all records in df_monthly having TTV='test', preserving the same columns. 
And set df_validate = all records in df_monthly having TTV='validate', preserving the same columns. 
And set df_forecast = all records in df_monthly having TTV='forecast', preserving the same columns. 
How many records are in df_train, df_test, df_validate, df_forecast?

Show 5 random records from df_train


## 2. train model

Use Tensorflow to train a simple Multilayer Perceptron model having 1 hidden layer.
That model's inputs will be the year, month, ward, primary_type, delta_count, count_0 columns in df_test.
And that model's outputs will be the count_1, count_2, count_3, count_4 columns.
Use your best guess about the number of neurons to use in that hidden layer, 
choose whatever seems most appropriate for this number of inputs and outputs.
Use df_test as the testing sample of data.
Name that trained model nnet.
Save the nnet model in folder 'models'


## 3. generate predictions

Use the nnet model to generate predictions on df_test, and append those predictions to
df_test as new columns called 'count_1_pred' etc

Then do the same for df_validate.

Then do the same for df_forecast.

Show 5 random records from df_validate


## 4. Create a dashboard of model validation tables and plots:

Plot 1: Start with df_monthly, then groupby date, TTV and compute sum(count_0) as total_count.
Then sum the train + test timeseries and plot that sum versus date.
Then plot the validation timeseries versus date.
And plot the forecast timeseries versus date.
Use connected scatterplot and color code by TTV.

Table 1: Then use the count_1 and count_1_pred columns in df_validate to calculate the net model's MAE and RMSE and R2 validation scores
Then do the same using columns count_2, count_3, count_4 
Show results in a table

plot 2: use df_validate to show timeseries plots of summed count_0 for all THEFT records versus date.
Also put vertical errorbars on the count_0 curve, assume each errorbar extends up/down by sqrt(count_0).
Color the count_0 curve blue.
Then overplot summed count_1_pred versus date + 1 month
Then overplot summed count_2_pred versus date + 2 month
Then overplot summed count_3_pred versus date + 3 month
Then overplot summed count_4_pred versus date + 4 month

plot 3: use df_validate to show timeseries plots of summed count_0 for all primary_type for wards 27, 29, and 38 vesus date.
Also put vertical errorbars on the count_0 curve, assume each errorbar extends up/down by sqrt(count_0)
Then overplot summed count_1_pred versus date + 1 month
Then overplot summed count_2_pred versus date + 2 month
Then overplot summed count_3_pred versus date + 3 month
Then overplot summed count_4_pred versus date + 4 month
Use red for ward 27 plots.
Use blue for ward 29 plots.
Use green for ward 38 plots.
The plot's vertical axis should use logarithmic axis.
Add legend to upper right corner of this plot
Show as connected scatterplots

Stack each plot or table vertically.
Use 20px of margin/padding between plots and tables. Set gap, margin, and padding to minimal values throughout.
Create a distinct, self-contained legend for each individual plot. Do not share or consolidate legends across plots. 
Set the vertical space between adjacent tables equal to 20px.
Also set the vertical space between a table and a plot at 20px.
Every chart must have its own legend embedded within it.
Store that dashboard as an html file named nnet_dashboard.html


