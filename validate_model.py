#!/usr/bin/env python3
# validate_model.py
# Load the trained skforecast model, run multi-horizon rolling predictions on the
# validate/forecast splits, and render an HTML dashboard of validation tables and plots.

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- Load data ----------
df_monthly = pd.read_csv("data/crimes_monthly.csv", parse_dates=["date"], low_memory=False)

KEEP = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]
df_train    = df_monthly.loc[df_monthly["TTV"] == "train",    KEEP].reset_index(drop=True)
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", KEEP].reset_index(drop=True)
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", KEEP].reset_index(drop=True)

print(f"df_train    : {len(df_train):,} records")
print(f"df_validate : {len(df_validate):,} records")
print(f"df_forecast : {len(df_forecast):,} records")


# ---------- Load forecaster ----------
forecaster = joblib.load("models/forecaster.joblib")
print(f"\nLoaded forecaster: {type(forecaster).__name__}")
print(f"  series seen during fit: {len(forecaster.series_names_in_)}")
print(f"  lags: {list(forecaster.lags)}   exog features: {forecaster.exog_names_in_}")


# ---------- Rolling-origin prediction over validate + forecast dates ----------
# Strategy: refit=False. The forecaster was already fit on 2022-05..2024-12. For each
# origin date t in the validate/forecast window we build a last_window of the 6 most
# recent observed count_0 values ending at t (using the full observed panel), call
# predict(steps=4) with future year/month exog, and record the per-horizon predictions.
# Refit each month would retrain XGBoost 15x — unnecessary given the stable feature set.

all_data = df_monthly.sort_values(["ward", "primary_type", "date"]).copy()
all_data["series_id"] = all_data["ward"].astype(str) + "_" + all_data["primary_type"]
series_full = all_data.pivot(index="date", columns="series_id", values="count_0")
series_full.index.freq = "MS"

exog_dates = pd.date_range(
    series_full.index.min(),
    series_full.index.max() + pd.offsets.MonthBegin(4),
    freq="MS",
)
full_exog = pd.DataFrame(
    {"year": exog_dates.year, "month": exog_dates.month},
    index=exog_dates,
)
full_exog.index.freq = "MS"

origin_dates = sorted(set(df_validate["date"]) | set(df_forecast["date"]))
print(f"\nPredicting 4 horizons at {len(origin_dates)} origins "
      f"({origin_dates[0].date()} -> {origin_dates[-1].date()})...")

all_preds = []
for t in origin_dates:
    last_window = series_full.loc[:t].tail(6).astype(float)
    future_start = t + pd.offsets.MonthBegin(1)
    future_end = t + pd.offsets.MonthBegin(4)
    future_exog = full_exog.loc[future_start:future_end]
    preds = forecaster.predict(steps=4, last_window=last_window, exog=future_exog)
    preds = preds.reset_index()
    preds.columns = ["pred_date" if c in (None, "index") else c for c in preds.columns]
    preds["origin_date"] = t
    preds["horizon"] = (
        (preds["pred_date"].dt.year - t.year) * 12 + (preds["pred_date"].dt.month - t.month)
    ).astype(int)
    all_preds.append(preds)

pred_long = pd.concat(all_preds, ignore_index=True)
print(f"Collected {len(pred_long):,} prediction rows (long format)")

pred_wide = pred_long.pivot_table(
    index=["origin_date", "level"],
    columns="horizon",
    values="pred",
).reset_index()
pred_wide.columns.name = None
pred_wide = pred_wide.rename(
    columns={
        "origin_date": "date",
        "level": "series_id",
        1: "count_1_pred",
        2: "count_2_pred",
        3: "count_3_pred",
        4: "count_4_pred",
    }
)
split = pred_wide["series_id"].str.split("_", n=1, expand=True)
pred_wide["ward"] = split[0].astype(int)
pred_wide["primary_type"] = split[1]
pred_cols = ["date", "ward", "primary_type",
             "count_1_pred", "count_2_pred", "count_3_pred", "count_4_pred"]
pred_wide = pred_wide[pred_cols]

df_validate = df_validate.merge(pred_wide, on=["date", "ward", "primary_type"], how="left")
df_forecast = df_forecast.merge(pred_wide, on=["date", "ward", "primary_type"], how="left")
print(f"\ndf_validate shape after merge: {df_validate.shape}")
print(f"df_forecast shape after merge: {df_forecast.shape}")


# ---------- Required sanity prints ----------
print("\nAll records in df_validate having primary_type=THEFT, ward=27:")
sub_v = df_validate[(df_validate["primary_type"] == "THEFT") & (df_validate["ward"] == 27)]
with pd.option_context("display.max_rows", None, "display.max_columns", None,
                       "display.width", None, "display.expand_frame_repr", False):
    print(sub_v.to_string(index=False))
print("\nAll records in df_forecast having primary_type=ARSON, ward=27:")
sub_f = df_forecast[(df_forecast["primary_type"] == "ARSON") & (df_forecast["ward"] == 27)]
with pd.option_context("display.max_rows", None, "display.max_columns", None,
                       "display.width", None, "display.expand_frame_repr", False):
    print(sub_f.to_string(index=False))


# ---------- Dashboard ----------
print("\nBuilding dashboard...")
os.makedirs("docs", exist_ok=True)


def scores_row(actual, pred):
    mask = actual.notna() & pred.notna()
    a, p = actual[mask], pred[mask]
    return {
        "MAE":  mean_absolute_error(a, p),
        "RMSE": mean_squared_error(a, p) ** 0.5,
        "R2":   r2_score(a, p),
    }


# Plot 1 — total monthly count by TTV
agg = df_monthly.groupby(["date", "TTV"], as_index=False)["count_0"].sum().rename(
    columns={"count_0": "total_count"}
)
ttv_colors = {"train": "#1f77b4", "validate": "#2ca02c", "forecast": "#d62728",
              "incomplete": "#7f7f7f"}
fig1 = go.Figure()
for ttv in ["train", "validate", "forecast"]:
    seg = agg[agg["TTV"] == ttv].sort_values("date")
    fig1.add_trace(go.Scatter(
        x=seg["date"], y=seg["total_count"],
        mode="lines+markers", name=ttv,
        line=dict(color=ttv_colors[ttv]), marker=dict(color=ttv_colors[ttv]),
    ))
fig1.update_layout(
    title="Plot 1 — total monthly crime count by TTV",
    xaxis_title="date", yaxis_title="sum(count_0)",
    height=380, margin=dict(l=60, r=20, t=50, b=50),
    legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
)


# Table 1 — validation scores per horizon
rows = []
for k in (1, 2, 3, 4):
    s = scores_row(df_validate[f"count_{k}"], df_validate[f"count_{k}_pred"])
    rows.append({"horizon": f"count_{k}_pred vs count_{k}",
                 "MAE": round(s["MAE"], 3),
                 "RMSE": round(s["RMSE"], 3),
                 "R2": round(s["R2"], 3)})
scores_df = pd.DataFrame(rows)
tbl1 = go.Figure(go.Table(
    header=dict(values=list(scores_df.columns), fill_color="#eaeaea", align="left"),
    cells=dict(values=[scores_df[c] for c in scores_df.columns], align="left"),
))
tbl1.update_layout(title="Table 1 — validation scores",
                   height=240, margin=dict(l=20, r=20, t=50, b=10))


# Table 2 — feature importances (importance column truncated to first 5 chars, descending)
fi = forecaster.get_feature_importances().sort_values("importance", ascending=False).reset_index(drop=True)
fi_disp = fi.astype(str).copy()
fi_disp["importance"] = fi_disp["importance"].str[:5]
tbl2 = go.Figure(go.Table(
    header=dict(values=list(fi_disp.columns), fill_color="#eaeaea", align="left"),
    cells=dict(values=[fi_disp[c] for c in fi_disp.columns], align="left"),
))
tbl2.update_layout(title="Table 2 — feature importances (importance truncated to 5 chars)",
                   height=320, margin=dict(l=20, r=20, t=50, b=10))


# Plots 2..5 — scatter count_N_pred vs count_N, opaque blue dots
def scatter_pred_vs_actual(k):
    df = df_validate[[f"count_{k}", f"count_{k}_pred"]].copy()
    df = df[(df[f"count_{k}"] > 0) & (df[f"count_{k}_pred"] > 0)]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df[f"count_{k}"], y=df[f"count_{k}_pred"],
        mode="markers", name="validate",
        marker=dict(color="blue", opacity=0.4, size=5),
    ))
    xs = np.array([0.8, 600])
    fig.add_trace(go.Scatter(
        x=xs, y=xs, mode="lines", name="prediction=actual",
        line=dict(color="black", dash="dash"),
    ))
    fig.update_xaxes(type="log", range=[np.log10(0.8), np.log10(600)], title=f"count_{k} (actual)")
    fig.update_yaxes(type="log", range=[np.log10(0.2), np.log10(600)], title=f"count_{k}_pred (prediction)")
    fig.update_layout(
        title=f"Plot {k + 1} — count_{k}_pred vs count_{k}",
        height=420, margin=dict(l=70, r=20, t=50, b=60),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


fig2, fig3, fig4, fig5 = (scatter_pred_vs_actual(k) for k in (1, 2, 3, 4))


# Plots 6..8 — per-primary_type totals with multi-horizon forecasts
def per_type_timeseries(primary_type, plot_idx):
    sub = df_validate[df_validate["primary_type"] == primary_type].copy()
    g = sub.groupby("date", as_index=False).agg(
        count_0=("count_0", "sum"),
        count_1_pred=("count_1_pred", "sum"),
        count_2_pred=("count_2_pred", "sum"),
        count_3_pred=("count_3_pred", "sum"),
        count_4_pred=("count_4_pred", "sum"),
    ).sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["count_0"], mode="lines+markers",
        name=f"{primary_type} count_0",
        line=dict(color="blue"), marker=dict(color="blue"),
        error_y=dict(type="data", array=np.sqrt(g["count_0"]), visible=True, color="blue"),
    ))
    pred_colors = {"count_1_pred": "orange", "count_2_pred": "green",
                   "count_3_pred": "purple", "count_4_pred": "brown"}
    for k in (1, 2, 3, 4):
        col = f"count_{k}_pred"
        fig.add_trace(go.Scatter(
            x=g["date"] + pd.DateOffset(months=k), y=g[col],
            mode="lines+markers", name=col,
            line=dict(color=pred_colors[col]), marker=dict(color=pred_colors[col]),
        ))
    fig.update_layout(
        title=f"Plot {plot_idx} — {primary_type} timeseries with multi-horizon forecasts",
        xaxis_title="date", yaxis_title="sum(count_0)",
        height=420, margin=dict(l=70, r=20, t=50, b=60),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


fig6 = per_type_timeseries("THEFT", 6)
fig7 = per_type_timeseries("BURGLARY", 7)
fig8 = per_type_timeseries("ARSON", 8)


# Plot 9 — per-ward timeseries for wards 27, 29, 38 with multi-horizon forecasts, log y
ward_colors = {27: "red", 29: "blue", 38: "green"}
fig9 = go.Figure()
for ward, wcolor in ward_colors.items():
    sub = df_validate[df_validate["ward"] == ward].copy()
    g = sub.groupby("date", as_index=False).agg(
        count_0=("count_0", "sum"),
        count_1_pred=("count_1_pred", "sum"),
        count_2_pred=("count_2_pred", "sum"),
        count_3_pred=("count_3_pred", "sum"),
        count_4_pred=("count_4_pred", "sum"),
    ).sort_values("date")
    fig9.add_trace(go.Scatter(
        x=g["date"], y=g["count_0"], mode="lines+markers",
        name=f"ward {ward} count_0",
        line=dict(color=wcolor), marker=dict(color=wcolor, size=7),
        error_y=dict(type="data", array=np.sqrt(g["count_0"]), visible=True, color=wcolor),
    ))
    for k in (1, 2, 3, 4):
        col = f"count_{k}_pred"
        fig9.add_trace(go.Scatter(
            x=g["date"] + pd.DateOffset(months=k), y=g[col],
            mode="lines+markers", name=f"ward {ward} {col}",
            line=dict(color=wcolor, dash="dot"), marker=dict(color=wcolor, size=4),
            opacity=0.6,
        ))
fig9.update_yaxes(type="log")
fig9.update_layout(
    title="Plot 9 — per-ward timeseries with multi-horizon forecasts (wards 27, 29, 38)",
    xaxis_title="date", yaxis_title="sum(count_0) (log)",
    height=520, margin=dict(l=70, r=20, t=50, b=60),
    legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.7)"),
)


# Plot 10 — THEFT heatmap over a Chicago streetmap, most recent complete month
theft_all = pd.read_csv("data/crimes.csv",
                        usecols=["date", "primary_type", "latitude", "longitude"],
                        low_memory=False)
theft_all = theft_all[theft_all["primary_type"] == "THEFT"].copy()
theft_all["date"] = pd.to_datetime(theft_all["date"], errors="coerce")
theft_all = theft_all.dropna(subset=["date", "latitude", "longitude"])
# The latest month in the source file is known-incomplete — use the month before it
latest_period = theft_all["date"].dt.to_period("M").max()
last_complete = latest_period - 1
theft_recent = theft_all[theft_all["date"].dt.to_period("M") == last_complete]
print(f"Plot 10: THEFT in {last_complete} ({len(theft_recent):,} records)")

# Pre-bin on a regular lat/lon grid; pass each non-empty bin centre to Densitymapbox
# with the raw bin count (linear color scale per the current spec)
nx, ny = 90, 90
lon_edges = np.linspace(-87.85, -87.5, nx + 1)
lat_edges = np.linspace(41.65, 42.05, ny + 1)
H, _, _ = np.histogram2d(theft_recent["latitude"], theft_recent["longitude"],
                         bins=[lat_edges, lon_edges])
lat_ctr = 0.5 * (lat_edges[:-1] + lat_edges[1:])
lon_ctr = 0.5 * (lon_edges[:-1] + lon_edges[1:])
ii, jj = np.nonzero(H)
z = H[ii, jj]

fig10 = go.Figure(go.Densitymapbox(
    lat=lat_ctr[ii], lon=lon_ctr[jj], z=z,
    radius=12,
    colorscale="YlOrRd",
    colorbar=dict(title="count"),
))
fig10.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 41.85, "lon": -87.675},
    mapbox_zoom=10,
    title=f"Plot 10 — THEFT heatmap over Chicago ({last_complete})",
    height=600, margin=dict(l=0, r=0, t=50, b=0),
)


# ---------- Assemble dashboard: exactly 10 px between adjacent blocks ----------
css = """
body { margin: 0; padding: 0; font-family: sans-serif; }
.block { margin: 0; padding: 0; }
.block + .block { margin-top: 10px; }
img { display: block; max-width: 100%; height: auto; }
"""
html_parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Chicago crime — forecast dashboard</title>",
    f"<style>{css}</style></head><body>",
]
html_parts.append(
    f"<div class='block'>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
)
for fig in (tbl1, tbl2, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
    html_parts.append(
        f"<div class='block'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
    )
html_parts.append("</body></html>")

dashboard_path = "docs/forecast_dashboard.html"
with open(dashboard_path, "w") as f:
    f.write("".join(html_parts))
print(f"Saved dashboard to {dashboard_path}")
