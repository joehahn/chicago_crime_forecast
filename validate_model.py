"""
validate_model.py

Loads the trained skforecast model, generates 1-to-4-month-ahead forecasts at
every validate/forecast date for every (ward, primary_type) series, computes
validation scores, and renders the model-validation dashboard to
docs/forecast_dashboard.html.
"""

import os
from math import cos, radians

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skforecast.utils import load_forecaster


# ---------------------------------------------------------------------------
# Load data and model.
# ---------------------------------------------------------------------------

df_monthly = pd.read_csv("data/crimes_monthly.csv", parse_dates=["date"])

keep_cols = [
    "date", "year", "month", "ward", "primary_type",
    "delta_count", "count_0", "count_1", "count_2", "count_3", "count_4",
]
df_train = df_monthly.loc[df_monthly["TTV"] == "train", keep_cols].copy()
df_validate = df_monthly.loc[df_monthly["TTV"] == "validate", keep_cols].copy()
df_forecast = df_monthly.loc[df_monthly["TTV"] == "forecast", keep_cols].copy()

print(f"df_train    has {len(df_train):,} records")
print(f"df_validate has {len(df_validate):,} records")
print(f"df_forecast has {len(df_forecast):,} records")

forecaster = load_forecaster("models/forecaster.joblib", verbose=False)
print("loaded forecaster from models/forecaster.joblib")


# ---------------------------------------------------------------------------
# Build the full wide panel of actual count_0 for all TTV values except
# 'incomplete' (we treat the last month as not fully observed, but we still
# include it so it can be used as history for the preceding prediction).
# Indexed by date at monthly-start frequency; one column per series_id.
# ---------------------------------------------------------------------------

df_all = df_monthly.copy()
df_all["series_id"] = (
    "w" + df_all["ward"].astype(int).astype(str).str.zfill(2)
    + "_" + df_all["primary_type"].str.replace(" ", "_")
)

panel = (
    df_all.pivot_table(index="date", columns="series_id", values="count_0", aggfunc="first")
    .sort_index()
    .asfreq("MS")
)

# exog values for any month: year + month derived from the date
def exog_for_dates(dates):
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({"year": d.year, "month": d.month}, index=d)


# ---------------------------------------------------------------------------
# Prediction — for every validate/forecast date t, use the actual history
# ending at t as last_window and predict t+1..t+4. Strategy: reuse the
# single pre-trained forecaster (no refit), which is fast and sufficient
# given the compact validate window.
# ---------------------------------------------------------------------------

pred_dates = sorted(
    pd.concat([df_validate["date"], df_forecast["date"]]).unique()
)
print(f"\ngenerating 4-step forecasts at {len(pred_dates)} origin dates ...")

# accumulator: map (date, series_id) -> [pred_1, pred_2, pred_3, pred_4]
pred_store = {}

lags = 6
for t in pred_dates:
    t = pd.Timestamp(t)
    window = panel.loc[:t].tail(lags)
    future_dates = pd.date_range(
        start=t + pd.offsets.MonthBegin(1), periods=4, freq="MS"
    )
    exog_future = exog_for_dates(future_dates)
    preds = forecaster.predict(
        steps=4,
        last_window=window,
        exog=exog_future,
        suppress_warnings=True,
    )
    # preds is long-form: columns = ['level', 'pred'] with datetime index
    # normalize to wide: rows=future_dates, cols=series_id
    preds_wide = preds.reset_index().pivot_table(
        index=preds.index.name or "index",
        columns="level",
        values="pred",
        aggfunc="first",
    )
    preds_wide.index = pd.to_datetime(preds_wide.index)
    preds_wide = preds_wide.sort_index()
    for series_id in preds_wide.columns:
        row = preds_wide[series_id].tolist()  # 4 values for t+1..t+4
        pred_store[(t, series_id)] = row

print("forecasts complete")


# helper: attach count_N_pred columns to a dataframe
def attach_predictions(df):
    df = df.copy()
    df["series_id"] = (
        "w" + df["ward"].astype(int).astype(str).str.zfill(2)
        + "_" + df["primary_type"].str.replace(" ", "_")
    )
    for n in (1, 2, 3, 4):
        col = f"count_{n}_pred"
        df[col] = [
            pred_store.get((pd.Timestamp(d), sid), [np.nan] * 4)[n - 1]
            for d, sid in zip(df["date"], df["series_id"])
        ]
    return df


df_validate = attach_predictions(df_validate)
df_forecast = attach_predictions(df_forecast)


# ---------------------------------------------------------------------------
# Quick inspection prints.
# ---------------------------------------------------------------------------

print("\nAll THEFT records in df_validate where ward=27:")
print(
    df_validate[
        (df_validate["primary_type"] == "THEFT") & (df_validate["ward"] == 27)
    ].to_string()
)

print("\nAll ARSON records in df_forecast where ward=27:")
print(
    df_forecast[
        (df_forecast["primary_type"] == "ARSON") & (df_forecast["ward"] == 27)
    ].to_string()
)


# ---------------------------------------------------------------------------
# Build the dashboard.
# ---------------------------------------------------------------------------

os.makedirs("docs", exist_ok=True)
os.makedirs("docs/img", exist_ok=True)
figures_html = []  # list of HTML snippets stacked vertically in the dashboard


def add_plotly(fig, first=False):
    """Append a plotly figure to the dashboard, reusing the CDN-hosted runtime."""
    kwargs = dict(full_html=False, include_plotlyjs="cdn" if first else False)
    figures_html.append(f'<div class="block">{fig.to_html(**kwargs)}</div>')


def add_table(df, title=None):
    """Append a DataFrame as an HTML table."""
    parts = ['<div class="block">']
    if title:
        parts.append(f"<h3>{title}</h3>")
    parts.append(df.to_html(index=False, border=0, classes="tbl"))
    parts.append("</div>")
    figures_html.append("".join(parts))


# ------- Plot 1: total-count timeseries by TTV --------------------------------

totals = (
    df_monthly.groupby(["date", "TTV"])["count_0"].sum().rename("total_count").reset_index()
)
fig1 = go.Figure()
ttv_colors = {"train": "#1f77b4", "validate": "#ff7f0e", "forecast": "#2ca02c", "incomplete": "#888888"}
for ttv in ["train", "validate", "forecast"]:
    sub = totals[totals["TTV"] == ttv].sort_values("date")
    fig1.add_trace(
        go.Scatter(
            x=sub["date"], y=sub["total_count"], mode="lines+markers",
            name=ttv, line=dict(color=ttv_colors[ttv]),
        )
    )
fig1.update_layout(
    title="Plot 1 — Total monthly crime count, color-coded by TTV",
    xaxis_title="date", yaxis_title="total_count",
    showlegend=True, legend=dict(x=0.02, y=0.98),
    height=420, margin=dict(l=60, r=30, t=50, b=50),
)
add_plotly(fig1, first=True)


# ------- Table 1: validation scores -------------------------------------------

scores_rows = []
for n in (1, 2, 3, 4):
    actual_col, pred_col = f"count_{n}", f"count_{n}_pred"
    sub = df_validate[[actual_col, pred_col]].dropna()
    y_true = sub[actual_col].to_numpy()
    y_pred = sub[pred_col].to_numpy()
    scores_rows.append({
        "horizon": f"t+{n} months",
        "MAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
        "RMSE": f"{np.sqrt(mean_squared_error(y_true, y_pred)):.3f}",
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "n": len(sub),
    })
scores_df = pd.DataFrame(scores_rows)
print("\nvalidation scores:")
print(scores_df.to_string(index=False))
add_table(scores_df, title="Table 1 — Validation scores (df_validate)")


# ------- Table 2: feature importances -----------------------------------------

fi = forecaster.get_feature_importances().copy()
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
fi_disp = fi.astype(str)
fi_disp["importance"] = fi_disp["importance"].str.slice(0, 5)
add_table(fi_disp, title="Table 2 — Feature importances (descending)")


# ------- Plots 2-5: pred vs actual scatter ------------------------------------

def scatter_pred_vs_actual(n):
    col_a = f"count_{n}"
    col_p = f"count_{n}_pred"
    sub = df_validate[[col_a, col_p]].dropna()
    sub = sub[(sub[col_a] > 0) & (sub[col_p] > 0)]
    trace_cls = go.Scattergl if len(sub) > 1000 else go.Scatter
    fig = go.Figure()
    fig.add_trace(
        trace_cls(
            x=sub[col_a], y=sub[col_p], mode="markers",
            marker=dict(color="blue", opacity=0.4, size=5),
            name=f"count_{n}_pred",
        )
    )
    xs = np.array([0.8, 600])
    fig.add_trace(
        go.Scatter(
            x=xs, y=xs, mode="lines",
            line=dict(color="black", dash="dash"),
            name="prediction=actual",
        )
    )
    fig.update_layout(
        title=f"Plot {n+1} — count_{n}_pred vs. count_{n}",
        xaxis=dict(title=f"count_{n} (actual)", type="log", range=[np.log10(0.8), np.log10(600)]),
        yaxis=dict(title=f"count_{n}_pred", type="log", range=[np.log10(0.2), np.log10(600)]),
        showlegend=True,
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        height=460, margin=dict(l=60, r=30, t=50, b=50),
    )
    add_plotly(fig)


for n in (1, 2, 3, 4):
    scatter_pred_vs_actual(n)


# ------- Plots 6-8: primary_type timeseries with forecasts --------------------

def plot_primary_type_timeseries(ptype, plot_num, line_color="blue"):
    sub = df_validate[df_validate["primary_type"] == ptype].copy()
    by_date = sub.groupby("date").agg(
        count_0=("count_0", "sum"),
        count_1_pred=("count_1_pred", "sum"),
        count_2_pred=("count_2_pred", "sum"),
        count_3_pred=("count_3_pred", "sum"),
        count_4_pred=("count_4_pred", "sum"),
    ).reset_index().sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=by_date["date"], y=by_date["count_0"], mode="lines+markers",
            name="count_0 (actual)", line=dict(color=line_color),
            error_y=dict(type="data", array=np.sqrt(by_date["count_0"].to_numpy()), visible=True),
        )
    )
    for n, color in zip((1, 2, 3, 4), ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]):
        shifted_dates = by_date["date"] + pd.offsets.MonthBegin(n)
        fig.add_trace(
            go.Scatter(
                x=shifted_dates, y=by_date[f"count_{n}_pred"], mode="lines+markers",
                name=f"count_{n}_pred (t+{n})", line=dict(color=color),
            )
        )
    fig.update_layout(
        title=f"Plot {plot_num} — {ptype} monthly totals with multi-horizon forecasts",
        xaxis_title="date", yaxis_title="summed count_0",
        showlegend=True,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        height=460, margin=dict(l=60, r=30, t=50, b=50),
    )
    add_plotly(fig)


plot_primary_type_timeseries("THEFT", 6)
plot_primary_type_timeseries("BURGLARY", 7)
plot_primary_type_timeseries("ARSON", 8)


# ------- Plot 9: per-ward timeseries for wards 27, 29, 38 ---------------------

wards_to_plot = [27, 29, 38]
ward_colors = {27: "red", 29: "blue", 38: "green"}

fig9 = go.Figure()
for w in wards_to_plot:
    sub = df_validate[df_validate["ward"] == w].copy()
    by_date = sub.groupby("date").agg(
        count_0=("count_0", "sum"),
        count_1_pred=("count_1_pred", "sum"),
        count_2_pred=("count_2_pred", "sum"),
        count_3_pred=("count_3_pred", "sum"),
        count_4_pred=("count_4_pred", "sum"),
    ).reset_index().sort_values("date")
    c = ward_colors[w]
    fig9.add_trace(
        go.Scatter(
            x=by_date["date"], y=by_date["count_0"], mode="lines+markers",
            name=f"ward {w} count_0", line=dict(color=c),
            error_y=dict(type="data", array=np.sqrt(by_date["count_0"].to_numpy()), visible=True),
        )
    )
    for n, dash in zip((1, 2, 3, 4), ["dot", "dash", "longdash", "dashdot"]):
        shifted = by_date["date"] + pd.offsets.MonthBegin(n)
        fig9.add_trace(
            go.Scatter(
                x=shifted, y=by_date[f"count_{n}_pred"], mode="lines",
                name=f"ward {w} count_{n}_pred",
                line=dict(color=c, dash=dash),
            )
        )
fig9.update_layout(
    title="Plot 9 — Per-ward monthly totals with forecasts (wards 27, 29, 38)",
    xaxis_title="date", yaxis_title="summed count_0 (log)",
    yaxis_type="log",
    showlegend=True,
    legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
    height=540, margin=dict(l=60, r=30, t=50, b=50),
)
add_plotly(fig9)


# ------- Plot 10: THEFT heatmap over Chicago streetmap ------------------------

raw = pd.read_csv("data/crimes.csv", low_memory=False)
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw = raw.dropna(subset=["date", "latitude", "longitude"])

# most recent COMPLETE calendar month = the month before the latest recorded month
latest_ts = raw["date"].max()
latest_period = latest_ts.to_period("M")
recent_complete = (latest_period - 1).to_timestamp()  # first day of that month
month_end = (latest_period).to_timestamp() - pd.Timedelta(days=1)
# (recent_complete through the last day of the preceding month)
m_start = recent_complete
m_end = (recent_complete + pd.offsets.MonthEnd(0))

theft_month = raw[
    (raw["primary_type"] == "THEFT")
    & (raw["date"] >= m_start)
    & (raw["date"] <= m_end + pd.Timedelta(days=1))
].copy()
print(f"\nPlot 10 month: {m_start.date()}..{m_end.date()}  "
      f"({len(theft_month):,} THEFT records)")

fig10 = go.Figure(
    go.Densitymapbox(
        lat=theft_month["latitude"],
        lon=theft_month["longitude"],
        radius=8,
        colorscale="YlOrRd",
        showscale=True,
    )
)
fig10.update_layout(
    title=f"Plot 10 — THEFT heatmap over Chicago ({m_start.strftime('%Y-%m')})",
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=41.85, lon=-87.675),
        zoom=9.7,
    ),
    height=700, margin=dict(l=0, r=0, t=50, b=0),
)
add_plotly(fig10)


# ---------------------------------------------------------------------------
# Assemble the HTML dashboard — 10 px between every adjacent element.
# ---------------------------------------------------------------------------

dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chicago Crime — Forecast Validation Dashboard</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 10px; padding: 0; }}
  h1 {{ margin: 0 0 10px 0; padding: 0; }}
  h3 {{ margin: 0 0 6px 0; padding: 0; }}
  .block {{ margin: 0 0 10px 0; padding: 0; }}
  .block:last-child {{ margin-bottom: 0; }}
  table.tbl {{ border-collapse: collapse; }}
  table.tbl th, table.tbl td {{ padding: 4px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
  table.tbl th {{ background: #f4f4f4; text-align: left; }}
</style>
</head>
<body>
<div class="block"><h1>Chicago Crime — Forecast Validation Dashboard</h1></div>
{chr(10).join(figures_html)}
</body>
</html>
"""
out_path = "docs/forecast_dashboard.html"
with open(out_path, "w") as f:
    f.write(dashboard)
print(f"\nsaved dashboard to {out_path}")
