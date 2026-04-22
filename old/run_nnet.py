#!/usr/bin/env python3
"""Train a Keras MLP on Chicago monthly crime data and render a dashboard.

Reads  : data/crimes_monthly.csv
Writes : models/nnet.keras, models/nnet_preproc.npz
         docs/nnet_dashboard.html  (published via GitHub Pages)

Single multi-output regression: 6 inputs → 1 hidden dense layer → 4 outputs.
The dashboard is built as 4 independent Plotly figures stacked in an HTML
page with exactly 10 px of vertical margin between neighbors (enforced via
CSS). Each panel carries its own self-contained legend.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

SEED = 42
GAP_PX = 10
EPOCHS = 40
BATCH_SIZE = 128
HIDDEN_UNITS = 32  # 6 inputs, 4 outputs — ~2×(in+out) is a reasonable starting point

ROOT = Path(__file__).parent
MONTHLY_PATH = ROOT / "data" / "crimes_monthly.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "nnet.keras"
PREPROC_PATH = MODELS_DIR / "nnet_preproc.npz"
OUTPUT_PATH = ROOT / "docs" / "nnet_dashboard.html"

TRAIN_COLS = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]
FEATURES = ["year", "month", "ward", "primary_type", "delta_count", "count_0"]
TARGETS = ["count_1", "count_2", "count_3", "count_4"]

UPPER_RIGHT = dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                   bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1)

np.random.seed(SEED)
tf.random.set_seed(SEED)
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
# 2. Featurize, scale, and train the MLP
# ---------------------------------------------------------------------------
le = LabelEncoder().fit(df_monthly["primary_type"])

def featurize(df):
    d = df.copy()
    d["primary_type"] = le.transform(d["primary_type"])
    return d[FEATURES].astype(float).values

X_train = featurize(df_train)
X_test  = featurize(df_test)
y_train = df_train[TARGETS].astype(float).values
y_test  = df_test[TARGETS].astype(float).values

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_s = x_scaler.transform(X_train)
X_test_s  = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)
y_test_s  = y_scaler.transform(y_test)

nnet = keras.Sequential([
    keras.layers.Input(shape=(len(FEATURES),)),
    keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
    keras.layers.Dense(len(TARGETS), activation="linear"),
])
nnet.compile(optimizer="adam", loss="mse", metrics=["mae"])

print(f"\nTraining MLP ({len(FEATURES)} → {HIDDEN_UNITS} → {len(TARGETS)}) ...")
nnet.fit(X_train_s, y_train_s,
         validation_data=(X_test_s, y_test_s),
         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
nnet.save(MODEL_PATH)
np.savez(
    PREPROC_PATH,
    x_mean=x_scaler.mean_, x_scale=x_scaler.scale_,
    y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
    primary_types=np.array(list(le.classes_)),
)
print(f"Saved {MODEL_PATH}")
print(f"Saved {PREPROC_PATH}")


# ---------------------------------------------------------------------------
# 3. Predict on df_test, df_validate, df_forecast — no CSVs written
# ---------------------------------------------------------------------------
def add_predictions(df):
    X = featurize(df)
    ys = nnet.predict(x_scaler.transform(X), verbose=0)
    y  = y_scaler.inverse_transform(ys)
    for i, target in enumerate(TARGETS):
        n = target.split("_")[1]
        df[f"count_{n}_pred"] = y[:, i]
    return df

df_test     = add_predictions(df_test)
df_validate = add_predictions(df_validate)
df_forecast = add_predictions(df_forecast)

print("\n--- 5 random records from df_validate ---")
print(df_validate.sample(5, random_state=SEED).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Dashboard panels
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
train_test_ts = (
    df_monthly[df_monthly["TTV"].isin(["train", "test"])]
    .groupby("date")["count_0"].sum().reset_index(name="total_count")
)
fig1 = go.Figure(layout=base_layout("Plot 1 — Total crime count vs. date, by TTV split", 400))
fig1.add_trace(go.Scatter(x=train_test_ts["date"], y=train_test_ts["total_count"],
                          mode="lines+markers", name="train + test",
                          line=dict(color="steelblue")))
for label, color in [("validate", "orange"), ("forecast", "crimson")]:
    sub = by_ttv[by_ttv["TTV"] == label]
    fig1.add_trace(go.Scatter(x=sub["date"], y=sub["total_count"],
                              mode="lines+markers", name=label,
                              line=dict(color=color)))


# Table 1 — validation MAE / RMSE / R²
rows = []
for target in TARGETS:
    n = target.split("_")[1]
    mask = df_validate[target].notna()
    y_true = df_validate.loc[mask, target]
    y_pred = df_validate.loc[mask, f"count_{n}_pred"]
    rows.append({
        "target": target,
        "n":      int(mask.sum()),
        "MAE":    f"{mean_absolute_error(y_true, y_pred):.4f}",
        "RMSE":   f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}",
        "R²":     f"{r2_score(y_true, y_pred):.4f}",
    })
val_df = pd.DataFrame(rows)
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


# Plot 3 — per-ward timeseries with forecasts (27, 29, 38), log y
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
# 5. Assemble dashboard HTML with 10 px between panels
# ---------------------------------------------------------------------------
panels = [fig1, table1, fig2, fig3]

css = f"""
  body   {{ font-family: sans-serif; max-width: 1100px; margin: {GAP_PX}px auto; padding: 0 20px; }}
  h1     {{ margin: 0 0 {GAP_PX}px 0; }}
  .panel {{ margin: 0 0 {GAP_PX}px 0; padding: 0; }}
  .panel:last-child {{ margin-bottom: 0; }}
"""

body_parts = ["<h1>Chicago Crime — Neural Net Validation Dashboard</h1>"]
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
    "<title>Chicago Crime — Neural Net Validation Dashboard</title>\n"
    f"<style>{css}</style>\n"
    "</head>\n"
    "<body>\n"
    + "\n".join(body_parts)
    + "\n</body>\n</html>\n"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(html)
print(f"\nSaved dashboard to {OUTPUT_PATH}")
