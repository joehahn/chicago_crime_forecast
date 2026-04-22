#!/usr/bin/env python3
"""Train an skforecast recursive multi-series forecaster on monthly Chicago crime data.

Reads  : data/crimes_monthly.csv
Writes : models/forecaster.joblib
         docs/forecast_dashboard.html  (published via GitHub Pages)

Approach — one timeseries per (ward, primary_type), count_0 as the target.
Fit ONCE on all history before 2025-01-01 (TTV = 'train'). For each
validate/forecast date t, predict 4 months ahead using the stored model and
a last_window that contains the observed history up through t (no refit —
documented in forecast_model.md). That produces count_{1..4}_pred per row.

Dashboard is 4 independent Plotly figures stacked with exactly 10 px CSS
gaps between neighbors; each panel has its own self-contained legend.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
GAP_PX = 10
LAGS = 6
FORECAST_HORIZON = 4
CUTOFF_DATE = pd.Timestamp("2025-01-01")

ROOT = Path(__file__).parent
MONTHLY_PATH = ROOT / "data" / "crimes_monthly.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "forecaster.joblib"
OUTPUT_PATH = ROOT / "docs" / "forecast_dashboard.html"

TRAIN_COLS = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]

UPPER_RIGHT = dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                   bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------
print(f"Loading {MONTHLY_PATH} ...")
df_monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])

df_train    = df_monthly.loc[df_monthly["TTV"] == "train",    TRAIN_COLS].copy()
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", TRAIN_COLS].copy()
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", TRAIN_COLS].copy()

print(f"df_train    records: {len(df_train):,}")
print(f"df_validate records: {len(df_validate):,}")
print(f"df_forecast records: {len(df_forecast):,}")

print("\n--- 5 random records from df_train ---")
print(df_train.sample(5, random_state=SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Pivot to wide-format panel: rows=date, cols=(ward, primary_type) series
# ---------------------------------------------------------------------------
def series_name(ward, primary_type):
    # Keep ward as-is; slugify primary_type so column names are string-safe.
    pt = primary_type.replace(" ", "_").replace("-", "_")
    return f"w{int(ward)}__{pt}"

df_monthly["series"] = df_monthly.apply(
    lambda r: series_name(r["ward"], r["primary_type"]), axis=1,
)
panel = (
    df_monthly
    .pivot_table(index="date", columns="series", values="count_0", aggfunc="first")
    .sort_index()
    .asfreq("MS")
)
print(f"\nPanel shape: {panel.shape[0]} months × {panel.shape[1]} series")

# Shared exogenous features — year, month per date.
exog = pd.DataFrame({"year": panel.index.year, "month": panel.index.month}, index=panel.index)

# Fit window: everything strictly before CUTOFF_DATE (TTV = 'train').
fit_mask = panel.index < CUTOFF_DATE
panel_fit = panel.loc[fit_mask]
exog_fit  = exog.loc[fit_mask]
print(f"Fitting on {len(panel_fit)} months of history ({panel_fit.index.min().date()} "
      f"to {panel_fit.index.max().date()})")


# ---------------------------------------------------------------------------
# 3. Fit the ForecasterRecursiveMultiSeries (single fit, no later refit)
# ---------------------------------------------------------------------------
regressor = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1,
)
forecaster = ForecasterRecursiveMultiSeries(
    regressor=regressor,
    lags=LAGS,
    encoding="ordinal",  # attaches an integer series ID to each row of features
    transformer_series=None,
    transformer_exog=None,
)

print("\nFitting ForecasterRecursiveMultiSeries (XGBoost, 6 lags, year+month exog) ...")
forecaster.fit(series=panel_fit, exog=exog_fit, suppress_warnings=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(forecaster, MODEL_PATH)
print(f"Saved {MODEL_PATH}")


# ---------------------------------------------------------------------------
# 4. Walking-forward prediction across validate + forecast dates
#    For each date t, predict count_0 at t+1..t+4 using observed history ≤ t.
# ---------------------------------------------------------------------------
pred_dates = sorted(set(df_validate["date"]).union(set(df_forecast["date"])))
print(f"\nPredicting at {len(pred_dates)} validate/forecast dates "
      f"({pred_dates[0].date()} to {pred_dates[-1].date()}) ...")

# preds[(t, series_name)] = [c1_pred, c2_pred, c3_pred, c4_pred]
preds: dict = {}
for t in pred_dates:
    history = panel.loc[panel.index <= t]
    future_dates = pd.date_range(t + pd.DateOffset(months=1),
                                 periods=FORECAST_HORIZON, freq="MS")
    future_exog = pd.DataFrame(
        {"year": future_dates.year, "month": future_dates.month},
        index=future_dates,
    )
    pred = forecaster.predict(
        steps=FORECAST_HORIZON,
        last_window=history,
        exog=future_exog,
        suppress_warnings=True,
    )
    # skforecast 0.18 returns a long frame with columns ['level', 'pred'] indexed by date.
    # Normalize to a wide DataFrame: index=future_dates, columns=series_name.
    if {"level", "pred"}.issubset(pred.columns):
        wide = pred.pivot_table(index=pred.index, columns="level", values="pred")
    else:
        wide = pred  # already wide
    for sname in wide.columns:
        preds[(t, sname)] = wide[sname].tolist()

def attach_predictions(df):
    df = df.copy()
    for n in range(1, FORECAST_HORIZON + 1):
        df[f"count_{n}_pred"] = np.nan
    for idx, row in df.iterrows():
        key = (row["date"], series_name(row["ward"], row["primary_type"]))
        p = preds.get(key)
        if p is not None:
            for n in range(1, FORECAST_HORIZON + 1):
                df.at[idx, f"count_{n}_pred"] = p[n - 1]
    return df

df_validate = attach_predictions(df_validate)
df_forecast = attach_predictions(df_forecast)

print("\n--- df_validate: primary_type=THEFT, ward=27 ---")
print(df_validate[(df_validate["primary_type"] == "THEFT") & (df_validate["ward"] == 27)].to_string(index=False))
print("\n--- df_forecast: primary_type=THEFT, ward=27 ---")
print(df_forecast[(df_forecast["primary_type"] == "THEFT") & (df_forecast["ward"] == 27)].to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Dashboard panels
# ---------------------------------------------------------------------------
def base_layout(title, height, **kwargs):
    return dict(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=50),
        legend=UPPER_RIGHT,
        **kwargs,
    )


# Plot 1 — TTV timeseries
by_ttv = df_monthly.groupby(["date", "TTV"])["count_0"].sum().reset_index(name="total_count")
fig1 = go.Figure(layout=base_layout("Plot 1 — Total crime count vs. date, by TTV split", 400))
for label, color in [("train", "steelblue"), ("validate", "orange"), ("forecast", "crimson")]:
    sub = by_ttv[by_ttv["TTV"] == label]
    fig1.add_trace(go.Scatter(x=sub["date"], y=sub["total_count"],
                              mode="lines+markers", name=label,
                              line=dict(color=color)))


# Table 1 — validation MAE / RMSE / R²
rows = []
for n in range(1, FORECAST_HORIZON + 1):
    target, pred_col = f"count_{n}", f"count_{n}_pred"
    mask = df_validate[target].notna() & df_validate[pred_col].notna()
    y_true = df_validate.loc[mask, target]
    y_pred = df_validate.loc[mask, pred_col]
    rows.append({
        "target": target,
        "n":      int(mask.sum()),
        "MAE":    f"{mean_absolute_error(y_true, y_pred):.4f}",
        "RMSE":   f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}",
        "R²":     f"{r2_score(y_true, y_pred):.4f}",
    })
val_df = pd.DataFrame(rows)
print("\n--- Validation scores ---")
print(val_df.to_string(index=False))

table1 = go.Figure(data=[go.Table(
    header=dict(values=list(val_df.columns), fill_color="#e6eef7",
                align="left", font=dict(size=13)),
    cells=dict(values=[val_df[c] for c in val_df.columns],
               align="left", font=dict(size=12)),
)])
table1.update_layout(title="Table 1 — Validation scores on df_validate",
                     height=200, margin=dict(l=60, r=40, t=50, b=10))


# Plot 2 — THEFT timeseries with multi-horizon forecasts
theft = df_validate[df_validate["primary_type"] == "THEFT"]
theft_actual = theft.groupby("date")["count_0"].sum().reset_index()
fig2 = go.Figure(layout=base_layout("Plot 2 — THEFT timeseries with multi-horizon forecasts", 420))
fig2.add_trace(go.Scatter(
    x=theft_actual["date"], y=theft_actual["count_0"],
    mode="lines+markers", name="THEFT count_0 (actual)",
    line=dict(color="blue", width=2),
    error_y=dict(type="data", array=np.sqrt(theft_actual["count_0"]), visible=True),
))
HORIZON_COLORS = {1: "darkorange", 2: "seagreen", 3: "crimson", 4: "purple"}
for n in (1, 2, 3, 4):
    col = f"count_{n}_pred"
    pred = theft.groupby("date")[col].sum().reset_index()
    pred["shifted"] = pred["date"] + pd.DateOffset(months=n)
    fig2.add_trace(go.Scatter(
        x=pred["shifted"], y=pred[col],
        mode="lines+markers", name=f"{col} → date + {n} mo",
        line=dict(color=HORIZON_COLORS[n], width=1),
    ))
fig2.update_xaxes(title_text="date")
fig2.update_yaxes(title_text="summed count_0")


# Plot 3 — per-ward timeseries (27, 29, 38), log y
fig3 = go.Figure(layout=base_layout(
    "Plot 3 — Per-ward timeseries with multi-horizon forecasts (wards 27, 29, 38)", 480,
))
ward_colors = {27: "red", 29: "blue", 38: "green"}
dashes = ["dot", "dash", "dashdot", "longdash"]
for ward, color in ward_colors.items():
    sub = df_validate[df_validate["ward"] == ward]
    actual = sub.groupby("date")["count_0"].sum().reset_index()
    fig3.add_trace(go.Scatter(
        x=actual["date"], y=actual["count_0"],
        mode="lines+markers", name=f"ward {ward} count_0",
        line=dict(color=color, width=2),
        error_y=dict(type="data", array=np.sqrt(actual["count_0"]), visible=True),
    ))
    for n in (1, 2, 3, 4):
        col = f"count_{n}_pred"
        pred = sub.groupby("date")[col].sum().reset_index()
        pred["shifted"] = pred["date"] + pd.DateOffset(months=n)
        fig3.add_trace(go.Scatter(
            x=pred["shifted"], y=pred[col],
            mode="lines+markers", name=f"ward {ward} {col}",
            line=dict(color=color, width=1, dash=dashes[n - 1]),
        ))
fig3.update_xaxes(title_text="date")
fig3.update_yaxes(type="log", title_text="summed count_0 (log)")


# ---------------------------------------------------------------------------
# 6. Assemble dashboard HTML with 10 px between panels
# ---------------------------------------------------------------------------
panels = [fig1, table1, fig2, fig3]

css = f"""
  body   {{ font-family: sans-serif; max-width: 1100px; margin: {GAP_PX}px auto; padding: 0 20px; }}
  h1     {{ margin: 0 0 {GAP_PX}px 0; }}
  .panel {{ margin: 0 0 {GAP_PX}px 0; padding: 0; }}
  .panel:last-child {{ margin-bottom: 0; }}
"""

body_parts = ["<h1>Chicago Crime — skforecast Validation Dashboard</h1>"]
for i, fig in enumerate(panels):
    include_js = "cdn" if i == 0 else False
    body_parts.append('<div class="panel">')
    body_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
    body_parts.append("</div>")

html = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '<meta charset="utf-8">\n'
    "<title>Chicago Crime — skforecast Validation Dashboard</title>\n"
    f"<style>{css}</style>\n"
    "</head>\n"
    "<body>\n"
    + "\n".join(body_parts)
    + "\n</body>\n</html>\n"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(html)
print(f"\nSaved dashboard to {OUTPUT_PATH}")
