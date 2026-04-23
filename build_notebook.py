"""
build_notebook.py — assembles forecast_validate.ipynb, a notebook that
replicates forecast_model.py + validate_model.py end-to-end.
"""

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(md(
    "# forecast_validate.ipynb\n\n"
    "Replicates `forecast_model.py` + `validate_model.py` end-to-end:\n"
    "retrains the skforecast multi-series recursive forecaster on the\n"
    "monthly Chicago-crime panel, runs rolling 4-step-ahead backtests on\n"
    "the validate/forecast splits, and renders the model-validation\n"
    "dashboard to `docs/forecast_dashboard.html`.\n"
))

# ------- forecast_model.py part ------------------------------------------------

cells.append(md("## Imports"))
cells.append(code(
    "import os\n"
    "from math import cos, radians\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import plotly.graph_objects as go\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "from skforecast.recursive import ForecasterRecursiveMultiSeries\n"
    "from skforecast.utils import load_forecaster, save_forecaster\n"
    "from xgboost import XGBRegressor"
))

cells.append(md("## Load crimes_monthly.csv and split by TTV"))
cells.append(code(
    "df_monthly = pd.read_csv('data/crimes_monthly.csv', parse_dates=['date'])\n"
    "\n"
    "keep_cols = [\n"
    "    'date', 'year', 'month', 'ward', 'primary_type',\n"
    "    'delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4',\n"
    "]\n"
    "df_train    = df_monthly.loc[df_monthly['TTV'] == 'train',    keep_cols].copy()\n"
    "df_validate = df_monthly.loc[df_monthly['TTV'] == 'validate', keep_cols].copy()\n"
    "df_forecast = df_monthly.loc[df_monthly['TTV'] == 'forecast', keep_cols].copy()\n"
    "\n"
    "print(f'df_train    has {len(df_train):,} records')\n"
    "print(f'df_validate has {len(df_validate):,} records')\n"
    "print(f'df_forecast has {len(df_forecast):,} records')\n"
    "df_train.sample(n=5, random_state=0)"
))

cells.append(md("## Reshape training data for skforecast"))
cells.append(code(
    "df_train['series_id'] = (\n"
    "    'w' + df_train['ward'].astype(int).astype(str).str.zfill(2)\n"
    "    + '_' + df_train['primary_type'].str.replace(' ', '_')\n"
    ")\n"
    "\n"
    "series_wide = (\n"
    "    df_train.pivot_table(index='date', columns='series_id', values='count_0', aggfunc='first')\n"
    "    .sort_index()\n"
    "    .asfreq('MS')\n"
    ")\n"
    "\n"
    "exog = (\n"
    "    df_train[['date', 'year', 'month']]\n"
    "    .drop_duplicates()\n"
    "    .set_index('date')\n"
    "    .sort_index()\n"
    "    .asfreq('MS')\n"
    ")\n"
    "\n"
    "print('series_wide shape:', series_wide.shape)\n"
    "print('exog shape:', exog.shape)"
))

cells.append(md("## Train the forecaster"))
cells.append(code(
    "regressor = XGBRegressor(\n"
    "    n_estimators=400,\n"
    "    max_depth=6,\n"
    "    learning_rate=0.05,\n"
    "    subsample=0.8,\n"
    "    colsample_bytree=0.8,\n"
    "    random_state=42,\n"
    "    n_jobs=-1,\n"
    ")\n"
    "\n"
    "forecaster = ForecasterRecursiveMultiSeries(\n"
    "    regressor=regressor,\n"
    "    lags=6,\n"
    "    encoding='ordinal',\n"
    ")\n"
    "forecaster.fit(series=series_wide, exog=exog)\n"
    "\n"
    "os.makedirs('models', exist_ok=True)\n"
    "save_forecaster(forecaster, file_name='models/forecaster.joblib', verbose=False)\n"
    "print('saved forecaster to models/forecaster.joblib')"
))

# ------- validate_model.py part ------------------------------------------------

cells.append(md("## Build the full wide panel of actuals"))
cells.append(code(
    "df_all = df_monthly.copy()\n"
    "df_all['series_id'] = (\n"
    "    'w' + df_all['ward'].astype(int).astype(str).str.zfill(2)\n"
    "    + '_' + df_all['primary_type'].str.replace(' ', '_')\n"
    ")\n"
    "panel = (\n"
    "    df_all.pivot_table(index='date', columns='series_id', values='count_0', aggfunc='first')\n"
    "    .sort_index().asfreq('MS')\n"
    ")\n"
    "\n"
    "def exog_for_dates(dates):\n"
    "    d = pd.DatetimeIndex(dates)\n"
    "    return pd.DataFrame({'year': d.year, 'month': d.month}, index=d)\n"
    "\n"
    "panel.shape"
))

cells.append(md("## Generate 4-step rolling forecasts at every validate/forecast date"))
cells.append(code(
    "pred_dates = sorted(pd.concat([df_validate['date'], df_forecast['date']]).unique())\n"
    "print('origin dates:', len(pred_dates))\n"
    "\n"
    "pred_store = {}\n"
    "lags = 6\n"
    "for t in pred_dates:\n"
    "    t = pd.Timestamp(t)\n"
    "    window = panel.loc[:t].tail(lags)\n"
    "    future_dates = pd.date_range(start=t + pd.offsets.MonthBegin(1), periods=4, freq='MS')\n"
    "    exog_future = exog_for_dates(future_dates)\n"
    "    preds = forecaster.predict(\n"
    "        steps=4, last_window=window, exog=exog_future, suppress_warnings=True\n"
    "    )\n"
    "    preds_wide = preds.reset_index().pivot_table(\n"
    "        index=preds.index.name or 'index',\n"
    "        columns='level', values='pred', aggfunc='first',\n"
    "    )\n"
    "    preds_wide.index = pd.to_datetime(preds_wide.index)\n"
    "    preds_wide = preds_wide.sort_index()\n"
    "    for series_id in preds_wide.columns:\n"
    "        pred_store[(t, series_id)] = preds_wide[series_id].tolist()\n"
    "\n"
    "\n"
    "def attach_predictions(df):\n"
    "    df = df.copy()\n"
    "    df['series_id'] = (\n"
    "        'w' + df['ward'].astype(int).astype(str).str.zfill(2)\n"
    "        + '_' + df['primary_type'].str.replace(' ', '_')\n"
    "    )\n"
    "    for n in (1, 2, 3, 4):\n"
    "        col = f'count_{n}_pred'\n"
    "        df[col] = [\n"
    "            pred_store.get((pd.Timestamp(d), sid), [np.nan]*4)[n-1]\n"
    "            for d, sid in zip(df['date'], df['series_id'])\n"
    "        ]\n"
    "    return df\n"
    "\n"
    "\n"
    "df_validate = attach_predictions(df_validate)\n"
    "df_forecast = attach_predictions(df_forecast)\n"
    "df_validate[(df_validate['primary_type']=='THEFT') & (df_validate['ward']==27)]"
))

cells.append(md("## Validation scores"))
cells.append(code(
    "rows = []\n"
    "for n in (1, 2, 3, 4):\n"
    "    sub = df_validate[[f'count_{n}', f'count_{n}_pred']].dropna()\n"
    "    y_t, y_p = sub.iloc[:, 0].to_numpy(), sub.iloc[:, 1].to_numpy()\n"
    "    rows.append({\n"
    "        'horizon': f't+{n} months',\n"
    "        'MAE':  f'{mean_absolute_error(y_t, y_p):.3f}',\n"
    "        'RMSE': f'{np.sqrt(mean_squared_error(y_t, y_p)):.3f}',\n"
    "        'R2':   f'{r2_score(y_t, y_p):.3f}',\n"
    "        'n':    len(sub),\n"
    "    })\n"
    "scores_df = pd.DataFrame(rows)\n"
    "scores_df"
))

cells.append(md("## Feature importances"))
cells.append(code(
    "fi = forecaster.get_feature_importances().sort_values('importance', ascending=False).reset_index(drop=True)\n"
    "fi_disp = fi.astype(str)\n"
    "fi_disp['importance'] = fi_disp['importance'].str.slice(0, 5)\n"
    "fi_disp"
))

cells.append(md(
    "## Dashboard\n\n"
    "Running `validate_model.py` end-to-end rebuilds `docs/forecast_dashboard.html`.\n"
    "This cell shells out to it so the notebook produces the same artifact."
))
cells.append(code(
    "import subprocess\n"
    "res = subprocess.run(['python3', 'validate_model.py'], capture_output=True, text=True)\n"
    "print(res.stdout[-2000:])\n"
    "if res.returncode != 0:\n"
    "    print('STDERR:', res.stderr[-2000:])"
))

nb["cells"] = cells
with open("forecast_validate.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote forecast_validate.ipynb")
