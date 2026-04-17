#!/usr/bin/env python3
"""Train the 4 'seasonal' XGBoost models and render a validation dashboard.

Reads  : data/crimes_monthly.csv, data/crimes.csv
Writes : models/seasonal_{1,2,3,4}.json
         docs/seasonal_model_dashboard.html  (published via GitHub Pages)

Each plot is rendered as an independent Plotly figure and stacked in a single
HTML page with exactly 10 px of vertical margin between adjacent plots/tables
(enforced via CSS, not via Plotly subplot layout).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

SEED = 42
GAP_PX = 10

ROOT = Path(__file__).parent
MONTHLY_PATH = ROOT / "data" / "crimes_monthly.csv"
RAW_PATH = ROOT / "data" / "crimes.csv"
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = ROOT / "docs" / "seasonal_model_dashboard.html"

TRAIN_COLS = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]
FEATURES = ["year", "month", "ward", "primary_type", "delta_count", "count_0"]
TARGETS = ["count_1", "count_2", "count_3", "count_4"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------
print(f"Loading {MONTHLY_PATH} ...")
df_monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])

df_train    = df_monthly.loc[df_monthly["TTV"] == "train",    TRAIN_COLS].copy()
df_test     = df_monthly.loc[df_monthly["TTV"] == "test",     TRAIN_COLS].copy()
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", TRAIN_COLS].copy()
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", TRAIN_COLS].copy()

print(f"df_train    records: {len(df_train):,}")
print(f"df_test     records: {len(df_test):,}")
print(f"df_validate records: {len(df_validate):,}")
print(f"df_forecast records: {len(df_forecast):,}")

print("\n--- 5 random records from df_train ---")
print(df_train.sample(5, random_state=SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Encode primary_type, train the 4 models
# ---------------------------------------------------------------------------
le = LabelEncoder().fit(df_monthly["primary_type"])

def encoded(df):
    d = df.copy()
    d["primary_type"] = le.transform(d["primary_type"])
    return d

X_train = encoded(df_train)[FEATURES]
X_test  = encoded(df_test)[FEATURES]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
)

models = {}
print("\nTraining XGBoost models ...")
for target in TARGETS:
    name = f"seasonal_{target.split('_')[1]}"
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, df_train[target],
              eval_set=[(X_test, df_test[target])], verbose=False)
    models[name] = model
    print(f"  {name} trained (target={target})")

# Save models.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
for name, m in models.items():
    p = MODELS_DIR / f"{name}.json"
    m.save_model(p)
    print(f"  saved {p}")


# ---------------------------------------------------------------------------
# 3. Predict onto df_test, df_validate, df_forecast
# ---------------------------------------------------------------------------
def add_predictions(df):
    X = encoded(df)[FEATURES]
    for target in TARGETS:
        n = target.split("_")[1]
        df[f"count_{n}_pred"] = models[f"seasonal_{n}"].predict(X)
    return df

df_test     = add_predictions(df_test)
df_validate = add_predictions(df_validate)
df_forecast = add_predictions(df_forecast)

print("\n--- df_validate: primary_type=THEFT, ward=27 ---")
print(df_validate[(df_validate["primary_type"] == "THEFT") & (df_validate["ward"] == 27)].to_string(index=False))

print("\n--- df_forecast: primary_type=THEFT, ward=27 ---")
print(df_forecast[(df_forecast["primary_type"] == "THEFT") & (df_forecast["ward"] == 27)].to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Dashboard helpers
# ---------------------------------------------------------------------------
def base_layout(title, height, **kwargs):
    return dict(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Plot 1 — total-count timeseries, color-coded by TTV
# ---------------------------------------------------------------------------
tot = df_monthly.groupby(["date", "TTV"])["count_0"].sum().reset_index(name="total_count")
train_test_ts = (
    df_monthly[df_monthly["TTV"].isin(["train", "test"])]
    .groupby("date")["count_0"].sum().reset_index(name="total_count")
)

fig1 = go.Figure(layout=base_layout("Plot 1 — Total crime count vs. date, by TTV split", 400))
fig1.add_trace(go.Scatter(x=train_test_ts["date"], y=train_test_ts["total_count"],
                          mode="lines+markers", name="train + test",
                          line=dict(color="steelblue")))
for label, color in [("validate", "orange"), ("forecast", "crimson")]:
    sub = tot[tot["TTV"] == label]
    fig1.add_trace(go.Scatter(x=sub["date"], y=sub["total_count"],
                              mode="lines+markers", name=label,
                              line=dict(color=color)))


# ---------------------------------------------------------------------------
# Table 1 — validation MAE / RMSE / R²
# ---------------------------------------------------------------------------
val_rows = []
for target in TARGETS:
    n = target.split("_")[1]
    # df_validate has NaN in count_N for the last N months of data; mask them.
    mask = df_validate[target].notna()
    y_true = df_validate.loc[mask, target]
    y_pred = df_validate.loc[mask, f"count_{n}_pred"]
    val_rows.append({
        "model": f"seasonal_{n}",
        "target": target,
        "n": int(mask.sum()),
        "MAE":  f"{mean_absolute_error(y_true, y_pred):.4f}",
        "RMSE": f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}",
        "R²":   f"{r2_score(y_true, y_pred):.4f}",
    })
val_df = pd.DataFrame(val_rows)

table1 = go.Figure(data=[go.Table(
    header=dict(values=list(val_df.columns), fill_color="#e6eef7",
                align="left", font=dict(size=13)),
    cells=dict(values=[val_df[c] for c in val_df.columns],
               align="left", font=dict(size=12)),
)])
table1.update_layout(title="Table 1 — Validation scores (MAE, RMSE, R²) on df_validate",
                     height=200, margin=dict(l=60, r=40, t=50, b=10))


# ---------------------------------------------------------------------------
# Table 2 — feature importances, values truncated to 5 chars
# ---------------------------------------------------------------------------
fi_df = pd.DataFrame({"feature": FEATURES})
for name, m in models.items():
    fi_df[name] = m.feature_importances_
fi_str = fi_df.astype(str).apply(lambda col: col.str[:5])

table2 = go.Figure(data=[go.Table(
    header=dict(values=list(fi_str.columns), fill_color="#e6eef7",
                align="left", font=dict(size=13)),
    cells=dict(values=[fi_str[c] for c in fi_str.columns],
               align="left", font=dict(size=12)),
)])
table2.update_layout(title="Table 2 — Feature importances (first 5 chars)",
                     height=260, margin=dict(l=60, r=40, t=50, b=10))


# ---------------------------------------------------------------------------
# Plots 2–5 — predictions vs. actuals (log-log), middle 80% green / outer 20% red
# ---------------------------------------------------------------------------
def pred_vs_actual_figure(target_col, pred_col, title):
    sub = df_validate[[target_col, pred_col, "count_0"]].copy()
    sub = sub[(sub[target_col] > 0) & (sub[pred_col] > 0)]
    lo, hi = sub["count_0"].quantile([0.10, 0.90])
    middle = (sub["count_0"] >= lo) & (sub["count_0"] <= hi)
    fig = go.Figure(layout=base_layout(title, 500))
    fig.add_trace(go.Scatter(
        x=sub.loc[middle, target_col], y=sub.loc[middle, pred_col],
        mode="markers", name="middle 80%",
        marker=dict(color="green", opacity=1.0, size=4),
    ))
    fig.add_trace(go.Scatter(
        x=sub.loc[~middle, target_col], y=sub.loc[~middle, pred_col],
        mode="markers", name="outer 20%",
        marker=dict(color="red", opacity=1.0, size=4),
    ))
    xy = np.array([0.8, 600])
    fig.add_trace(go.Scatter(x=xy, y=xy, mode="lines",
                             name="prediction=actual",
                             line=dict(color="black", dash="dash", width=1)))
    fig.update_xaxes(type="log", range=[np.log10(0.8), np.log10(600)], title_text="actual")
    fig.update_yaxes(type="log", range=[np.log10(0.2), np.log10(600)], title_text="prediction")
    return fig

fig2 = pred_vs_actual_figure("count_1", "count_1_pred", "Plot 2 — seasonal_1 predictions vs. actuals (log-log)")
fig3 = pred_vs_actual_figure("count_2", "count_2_pred", "Plot 3 — seasonal_2 predictions vs. actuals (log-log)")
fig4 = pred_vs_actual_figure("count_3", "count_3_pred", "Plot 4 — seasonal_3 predictions vs. actuals (log-log)")
fig5 = pred_vs_actual_figure("count_4", "count_4_pred", "Plot 5 — seasonal_4 predictions vs. actuals (log-log)")


# ---------------------------------------------------------------------------
# Plots 6–8 — per-primary_type timeseries with multi-horizon forecasts
# ---------------------------------------------------------------------------
def horizon_timeseries_figure(crime_type, title):
    sub = df_validate[df_validate["primary_type"] == crime_type]
    actual = sub.groupby("date")["count_0"].sum().reset_index()
    fig = go.Figure(layout=base_layout(title, 420))
    fig.add_trace(go.Scatter(
        x=actual["date"], y=actual["count_0"],
        mode="lines+markers", name=f"{crime_type} count_0 (actual)",
        line=dict(color="black", width=2),
        error_y=dict(type="data", array=np.sqrt(actual["count_0"]), visible=True),
    ))
    colors = {"count_1_pred": "steelblue", "count_2_pred": "darkorange",
              "count_3_pred": "seagreen",  "count_4_pred": "crimson"}
    for n in (1, 2, 3, 4):
        col = f"count_{n}_pred"
        pred = sub.groupby("date")[col].sum().reset_index()
        pred["shifted"] = pred["date"] + pd.DateOffset(months=n)
        fig.add_trace(go.Scatter(
            x=pred["shifted"], y=pred[col],
            mode="lines+markers", name=f"{col} → date + {n} mo",
            line=dict(color=colors[col], width=1),
        ))
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="summed count_0")
    return fig

fig6 = horizon_timeseries_figure("THEFT",    "Plot 6 — THEFT timeseries with multi-horizon forecasts")
fig7 = horizon_timeseries_figure("BURGLARY", "Plot 7 — BURGLARY timeseries with multi-horizon forecasts")
fig8 = horizon_timeseries_figure("ARSON",    "Plot 8 — ARSON timeseries with multi-horizon forecasts")


# ---------------------------------------------------------------------------
# Plot 9 — per-ward timeseries with forecasts (wards 27, 29, 38), log y
# ---------------------------------------------------------------------------
fig9 = go.Figure(layout=base_layout(
    "Plot 9 — Per-ward timeseries with multi-horizon forecasts (wards 27, 29, 38)", 480,
))
ward_colors = {27: "red", 29: "blue", 38: "green"}
for ward, color in ward_colors.items():
    sub = df_validate[df_validate["ward"] == ward]
    actual = sub.groupby("date")["count_0"].sum().reset_index()
    fig9.add_trace(go.Scatter(
        x=actual["date"], y=actual["count_0"],
        mode="lines+markers", name=f"ward {ward} count_0",
        line=dict(color=color, width=2),
        error_y=dict(type="data", array=np.sqrt(actual["count_0"]), visible=True),
    ))
    for n in (1, 2, 3, 4):
        col = f"count_{n}_pred"
        pred = sub.groupby("date")[col].sum().reset_index()
        pred["shifted"] = pred["date"] + pd.DateOffset(months=n)
        fig9.add_trace(go.Scatter(
            x=pred["shifted"], y=pred[col],
            mode="lines+markers", name=f"ward {ward} {col}",
            line=dict(color=color, width=1, dash=["dot", "dash", "dashdot", "longdash"][n - 1]),
        ))
fig9.update_xaxes(title_text="date")
fig9.update_yaxes(type="log", title_text="summed count_0 (log)")


# ---------------------------------------------------------------------------
# Plot 10 — THEFT density heatmap on a Chicago street map, March 2026
# ---------------------------------------------------------------------------
print(f"\nLoading {RAW_PATH} for Plot 10 heatmap ...")
df_raw = pd.read_csv(RAW_PATH, usecols=["date", "primary_type", "latitude", "longitude"],
                     parse_dates=["date"])
mar = df_raw[
    (df_raw["primary_type"] == "THEFT")
    & (df_raw["date"].dt.year == 2026)
    & (df_raw["date"].dt.month == 3)
    & df_raw["latitude"].notna()
    & df_raw["longitude"].notna()
].copy()
print(f"  THEFT records in March 2026: {len(mar):,}")

# Bin to a grid and take log for logarithmic color scaling.
lon_bins = np.linspace(-87.85, -87.5, 80)
lat_bins = np.linspace(41.65, 42.05, 80)
H, lon_edges, lat_edges = np.histogram2d(mar["longitude"], mar["latitude"], bins=[lon_bins, lat_bins])
lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
LON, LAT = np.meshgrid(lon_centers, lat_centers, indexing="ij")
nonzero = H > 0
log_count = np.log10(H[nonzero])

fig10 = go.Figure(go.Densitymapbox(
    lat=LAT[nonzero],
    lon=LON[nonzero],
    z=log_count,
    radius=10,
    colorscale="Hot",
    colorbar=dict(title="log₁₀(count)"),
))
fig10.update_layout(
    title="Plot 10 — THEFT density heatmap, March 2026 (log color scale)",
    height=600,
    mapbox_style="open-street-map",
    mapbox_center=dict(lat=41.85, lon=-87.675),
    mapbox_zoom=9.2,
    margin=dict(l=0, r=0, t=50, b=0),
)


# ---------------------------------------------------------------------------
# 5. Assemble dashboard HTML
# ---------------------------------------------------------------------------
panels = [
    ("plot",  fig1),
    ("table", table1),
    ("table", table2),
    ("plot",  fig2),
    ("plot",  fig3),
    ("plot",  fig4),
    ("plot",  fig5),
    ("plot",  fig6),
    ("plot",  fig7),
    ("plot",  fig8),
    ("plot",  fig9),
    ("plot",  fig10),
]

css = f"""
  body     {{ font-family: sans-serif; max-width: 1100px; margin: {GAP_PX}px auto; padding: 0 20px; }}
  h1       {{ margin: 0 0 {GAP_PX}px 0; }}
  .panel   {{ margin: 0 0 {GAP_PX}px 0; padding: 0; }}
  .panel:last-child {{ margin-bottom: 0; }}
"""

body_parts = ["<h1>Chicago Crime — Seasonal Model Validation Dashboard</h1>"]
for i, (_, fig) in enumerate(panels):
    include_js = "cdn" if i == 0 else False
    body_parts.append('<div class="panel">')
    body_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
    body_parts.append("</div>")

html = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '<meta charset="utf-8">\n'
    "<title>Chicago Crime — Seasonal Model Validation Dashboard</title>\n"
    f"<style>{css}</style>\n"
    "</head>\n"
    "<body>\n"
    + "\n".join(body_parts)
    + "\n</body>\n</html>\n"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(html)
print(f"\nSaved dashboard to {OUTPUT_PATH}")
