"""Automobile Sales Dashboard (Plotly Dash): recession vs. non-recession analysis.

Run:
    python app.py                         # uses DATA_PATH below
    DATA_PATH=/path/or/url.csv python app.py

DATA_PATH may be a local CSV path or a URL (anything pandas.read_csv accepts).
Open http://127.0.0.1:8050 after starting.

Key modelling rules (see the prompt that produced this file):
  * Row grain is Date x State x Vehicle_Type; the panel may be unbalanced.
  * Month-level macro columns (GDP, Consumer_Confidence, ...) are analysed on a
    one-row-per-month frame so months with more rows are not overweighted.
  * Recession vs. non-recession comparisons use per-record averages, never raw
    totals, because the two regimes span different numbers of months.
  * `City` actually holds US state names; they are mapped to USPS codes, not
    geocoded (geocoding "Georgia" would resolve to the country).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("auto_dashboard")

# =============================================================================
# Configuration and constants
# =============================================================================
DATA_PATH = os.environ.get("DATA_PATH", "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d51iMGfp_t0QpO30Lym-dw/automobile-sales.csv")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8050"))
DEBUG = os.environ.get("DASH_DEBUG", "0") == "1"
CURRENCY_SYMBOL = "$"

REQUIRED_COLUMNS = [
    "Date", "Year", "Month", "Recession", "Consumer_Confidence", "Seasonality_Weight",
    "Price", "Advertising_Expenditure", "Competition", "GDP", "Growth_Rate",
    "Unemployment_Rate", "Automobile_Sales", "Vehicle_Type", "City",
]
_CANONICAL = {c.lower(): c for c in REQUIRED_COLUMNS}  # 'unemployment_rate' -> 'Unemployment_Rate'
MACRO_COLUMNS = ["Consumer_Confidence", "GDP", "Growth_Rate", "Unemployment_Rate", "Competition"]
MACRO_CONSTANCY_THRESHOLD = 0.95
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PERIOD_RECESSION = "Recession"
PERIOD_NON = "Non-Recession"
PERIOD_ALL = "All"
PERIOD_ORDER = [PERIOD_RECESSION, PERIOD_NON]
PERIOD_LABELS = {1: PERIOD_RECESSION, 0: PERIOD_NON}
PERIOD_COLORS = {PERIOD_RECESSION: "#C0392B", PERIOD_NON: "#2E86C1"}
COLOR_PRIMARY = "#1F3A5F"
COLOR_ACCENT = "#E67E22"
VEHICLE_PALETTE = px.colors.qualitative.Safe

METRIC_AVG = "avg"
METRIC_TOTAL = "total"
ROLLING_OPTIONS = [(0, "None"), (3, "3 months"), (6, "6 months"), (12, "12 months")]

TEMPLATE = "plotly_white"
FONT_FAMILY = "Segoe UI, Helvetica, Arial, sans-serif"
BASE_FONT_SIZE = 12
TITLE_SIZE = 16
CHART_HEIGHT = 390
MAP_HEIGHT = 520
Y_HEADROOM = 1.25
MAX_BUBBLE = 22
MIN_POINTS_CORR = 3
TOP_N_STATES = 5
GRAPH_CONFIG = {"displaylogo": False}
NO_DATA_MSG = "No data for this selection"

US_STATE_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

Chart = tuple  # (go.Figure, insight caption)


# =============================================================================
# Data loading and preparation
# =============================================================================
@dataclass
class DataReport:
    """Data-quality notes collected while preparing the dataset."""

    warnings: list = field(default_factory=list)
    ad_repeat_share: float = 0.0

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)


@lru_cache(maxsize=2)
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV (path or URL), standardise column names, validate required columns."""
    raw = pd.read_csv(path)
    raw = raw.rename(columns=lambda c: _CANONICAL.get(str(c).strip().lower(), c))
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return raw


def _split_camel(label: str) -> str:
    """'SmallFamilyCar' -> 'Small Family Car'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)


def prepare_data(raw: pd.DataFrame) -> tuple:
    """Clean, validate and enrich the raw frame. Returns (dataframe, DataReport)."""
    rep = DataReport()
    df = raw.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("Date", "Month", "Vehicle_Type", "City")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Vehicle_Type", "City"):
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    n0 = len(df)
    df = df.dropna(subset=["Date", "Recession", "Automobile_Sales"])
    if len(df) < n0:
        rep.warn(f"Dropped {n0 - len(df)} rows missing Date, Recession or Automobile_Sales.")

    bad_flag = ~df["Recession"].isin([0, 1])
    if bad_flag.any():
        rep.warn(f"Dropped {int(bad_flag.sum())} rows where Recession is not 0/1.")
    df = df.loc[~bad_flag].copy()
    df["Recession"] = df["Recession"].astype(int)

    negative = df["Automobile_Sales"] < 0
    if negative.any():
        rep.warn(f"Dropped {int(negative.sum())} rows with negative Automobile_Sales.")
        df = df.loc[~negative].copy()

    n_dup = int(df.duplicated().sum())
    if n_dup:
        rep.warn(f"Removed {n_dup} fully identical duplicate rows.")
        df = df.drop_duplicates().copy()
    n_key = int(df.duplicated(subset=["Date", "City", "Vehicle_Type"]).sum())
    if n_key:
        rep.warn(f"{n_key} rows repeat a Date + State + Vehicle_Type key (kept, not deduplicated).")

    # Trust Date over the supplied Year / Month columns.
    derived_year = df["Date"].dt.year
    if int((df["Year"] != derived_year).sum()):
        rep.warn("Year disagreed with Date for some rows; Year was re-derived from Date.")
    df["Year"] = derived_year.astype(int)
    df["Month_Num"] = df["Date"].dt.month
    month_names = df["Month_Num"].map(dict(enumerate(MONTHS, start=1)))
    given = df["Month"].astype(str).str[:3].str.title()
    if int((given != month_names).sum()):
        rep.warn("Month text disagreed with Date for some rows; Month was re-derived from Date.")
    df["Month"] = pd.Categorical(month_names, categories=MONTHS, ordered=True)
    df["Quarter"] = df["Date"].dt.quarter

    # Recession is a property of the month; enforce one value per Date.
    mixed = df.groupby("Date")["Recession"].nunique()
    if int((mixed > 1).sum()):
        rep.warn(f"Recession differed within {int((mixed > 1).sum())} dates; the mode was used.")
        df["Recession"] = df.groupby("Date")["Recession"].transform(lambda s: s.mode().iat[0]).astype(int)
    df["Period"] = df["Recession"].map(PERIOD_LABELS)

    df["Vehicle_Label"] = df["Vehicle_Type"].map(_split_camel)
    df["State_Code"] = df["City"].str.lower().map(US_STATE_CODES)
    unmapped = sorted(df.loc[df["State_Code"].isna(), "City"].unique())
    if unmapped:
        rep.warn(f"Unmapped State/Region values (excluded from the map only): {unmapped}")

    for col in MACRO_COLUMNS:
        share = float((df.groupby("Date")[col].nunique(dropna=True) <= 1).mean())
        if share < MACRO_CONSTANCY_THRESHOLD:
            rep.warn(f"{col} is not month-level constant ({share:.0%} of dates); monthly means are used.")

    ad = df.dropna(subset=["Advertising_Expenditure"])
    if len(ad):
        rep.ad_repeat_share = float(
            ad.duplicated(subset=["Date", "City", "Advertising_Expenditure"], keep=False).mean()
        )
    if rep.ad_repeat_share > 0:
        rep.warn(
            f"{rep.ad_repeat_share:.0%} of rows share an Advertising_Expenditure value with another row "
            "in the same month and state; ad-spend totals may double count if it is a month-level budget."
        )

    return df.sort_values("Date", kind="stable").reset_index(drop=True), rep


def apply_filters(df: pd.DataFrame, years, vehicles, states) -> pd.DataFrame:
    """Apply Year / Vehicle / State filters. An empty selection yields an empty frame."""
    if not years or not vehicles or not states:
        return df.iloc[0:0]
    mask = df["Year"].isin(years) & df["Vehicle_Label"].isin(vehicles) & df["City"].isin(states)
    return df.loc[mask]


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Date: month-level macro variables plus aggregated sales/ad metrics."""
    agg = {
        "Recession": ("Recession", "max"),
        "Avg_Sales": ("Automobile_Sales", "mean"),
        "Total_Sales": ("Automobile_Sales", "sum"),
        "Avg_Ad": ("Advertising_Expenditure", "mean"),
        "Avg_Price": ("Price", "mean"),
        "Avg_Seasonality": ("Seasonality_Weight", "mean"),
        "Records": ("Automobile_Sales", "size"),
    }
    agg.update({c: (c, "mean") for c in MACRO_COLUMNS})
    m = df.groupby("Date").agg(**agg).reset_index()
    m["Period"] = m["Recession"].map(PERIOD_LABELS)
    return m


def recession_windows(m: pd.DataFrame) -> list:
    """Contiguous runs of Recession == 1 as (start, end) timestamps."""
    flag = m["Recession"].eq(1)
    run_id = flag.ne(flag.shift()).cumsum()
    return [
        (grp["Date"].min().to_period("M").to_timestamp(), grp["Date"].max())
        for _, grp in m[flag].groupby(run_id[flag])
    ]


# =============================================================================
# Small helpers
# =============================================================================
def pearson(x: pd.Series, y: pd.Series) -> tuple:
    """Pearson r and n on complete pairs; r is NaN when undefined."""
    d = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1).dropna()
    if len(d) < MIN_POINTS_CORR or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return float("nan"), len(d)
    return float(d.iloc[:, 0].corr(d.iloc[:, 1])), len(d)


def pct_change(new, old) -> float:
    """Percent change of `new` relative to `old` (NaN if undefined)."""
    if new is None or old is None or pd.isna(new) or pd.isna(old) or old == 0:
        return float("nan")
    return float((new - old) / old * 100.0)


def style(fig: go.Figure, title: str, xt: str = None, yt: str = None, height: int = CHART_HEIGHT) -> go.Figure:
    """Apply the shared look: template, font, margins, legend placement, axis titles."""
    fig.update_layout(
        template=TEMPLATE, height=height,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=TITLE_SIZE)),
        font=dict(family=FONT_FAMILY, size=BASE_FONT_SIZE),
        margin=dict(l=55, r=20, t=75, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, title_text=""),
    )
    if xt is not None:
        fig.update_xaxes(title_text=xt)
    if yt is not None:
        fig.update_yaxes(title_text=yt)
    return fig


def empty_chart(msg: str = NO_DATA_MSG) -> Chart:
    """Friendly placeholder figure for empty selections."""
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=16, color="#7F8C8D"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=TEMPLATE, height=CHART_HEIGHT, margin=dict(l=20, r=20, t=40, b=20))
    return fig, msg


def _scatter_ols(m: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure:
    """px.scatter with an OLS trendline; falls back to no trendline if it cannot be fitted."""
    try:
        return px.scatter(m, x=x, y=y, trendline="ols", **kwargs)
    except Exception as exc:  # too few points per group, missing statsmodels, ...
        log.info("Trendline skipped (%s)", exc)
        return px.scatter(m, x=x, y=y, **kwargs)


def _corr_by_period(m: pd.DataFrame, x: str, y: str) -> str:
    parts = []
    for p in PERIOD_ORDER:
        sub = m[m["Period"] == p]
        r, n = pearson(sub[x], sub[y])
        if n:
            parts.append(f"{p}: " + (f"r = {r:.2f}" if not np.isnan(r) else "r n/a") + f" (n = {n})")
    return "; ".join(parts) if parts else "Not enough data for a correlation."


# =============================================================================
# Chart builders. Each returns (figure, one-line insight caption).
# =============================================================================
def chart_sales_over_time(df: pd.DataFrame, metric: str, window: int) -> Chart:
    """1. Average (or total) automobile sales over time, optional rolling average."""
    if df.empty:
        return empty_chart()
    m = build_monthly(df)
    col, label = ("Avg_Sales", "Average sales per record") if metric == METRIC_AVG else ("Total_Sales", "Total sales")
    smoothed = bool(window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["Date"], y=m[col], mode="lines", name=label, opacity=0.4 if smoothed else 1.0,
        line=dict(color=COLOR_PRIMARY, width=1.6 if smoothed else 2.6),
        hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:,.1f}<extra></extra>"))
    if smoothed:
        fig.add_trace(go.Scatter(
            x=m["Date"], y=m[col].rolling(window, min_periods=1).mean(), mode="lines",
            name=f"{window}-month rolling average", line=dict(color=COLOR_ACCENT, width=3),
            hovertemplate="%{x|%b %Y}<br>Rolling avg: %{y:,.1f}<extra></extra>"))
    style(fig, f"{label} over time", "Date", label)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    hi, lo = m.loc[m[col].idxmax()], m.loc[m[col].idxmin()]
    return fig, (f"Peak {hi[col]:,.1f} in {hi['Date']:%b %Y}; low {lo[col]:,.1f} in {lo['Date']:%b %Y}.")


def chart_recession_windows(df: pd.DataFrame, metric: str) -> Chart:
    """2. Sales over time with recession windows shaded."""
    if df.empty:
        return empty_chart()
    m = build_monthly(df)
    col, label = ("Avg_Sales", "Average sales per record") if metric == METRIC_AVG else ("Total_Sales", "Total sales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["Date"], y=m[col], mode="lines", name=label,
                             line=dict(color=COLOR_PRIMARY, width=2),
                             hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:,.1f}<extra></extra>"))
    rec = m[m["Recession"] == 1]
    windows = recession_windows(m)
    for x0, x1 in windows:
        fig.add_vrect(x0=x0.strftime("%Y-%m-%d"), x1=x1.strftime("%Y-%m-%d"),
                      fillcolor=PERIOD_COLORS[PERIOD_RECESSION], opacity=0.15, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=rec["Date"], y=rec[col], mode="markers", name="Recession months",
                             marker=dict(color=PERIOD_COLORS[PERIOD_RECESSION], size=6),
                             hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:,.1f}<extra></extra>"))
    style(fig, f"{label} with recession periods shaded", "Date", label)
    if rec.empty:
        return fig, "No recession months in this selection."
    avg_in, out = rec[col].mean(), m.loc[m["Recession"] == 0, col].mean()
    note = f"{len(windows)} recession window(s); mean {avg_in:,.1f} inside"
    note += f" vs {out:,.1f} outside." if not np.isnan(out) else "."
    return fig, note


def chart_ad_vs_sales(df: pd.DataFrame) -> Chart:
    """3. Advertising expenditure vs. sales (monthly aggregates), by period."""
    m = build_monthly(df).dropna(subset=["Avg_Ad", "Avg_Sales"]) if not df.empty else pd.DataFrame()
    if m.empty:
        return empty_chart()
    fig = _scatter_ols(m, "Avg_Ad", "Avg_Sales", color="Period", color_discrete_map=PERIOD_COLORS,
                       category_orders={"Period": PERIOD_ORDER}, opacity=0.7,
                       hover_data={"Date": "|%b %Y"})
    style(fig, "Advertising expenditure vs. automobile sales (monthly)",
          "Average advertising expenditure per record", "Average sales per record")
    return fig, _corr_by_period(m, "Avg_Ad", "Avg_Sales")


def _period_bar(df: pd.DataFrame, col: str, title: str, ylabel: str, what: str) -> Chart:
    """Shared Recession vs. Non-Recession per-record average bar chart with % difference."""
    if df.empty:
        return empty_chart()
    g = df.groupby("Period")[col].mean().reindex(PERIOD_ORDER).dropna()
    if g.empty:
        return empty_chart()
    fig = go.Figure(go.Bar(
        x=list(g.index), y=g.values, marker_color=[PERIOD_COLORS[p] for p in g.index],
        text=[f"{v:,.1f}" for v in g.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>" + ylabel + ": %{y:,.1f}<extra></extra>"))
    fig.update_yaxes(range=[0, max(float(g.max()), 1.0) * Y_HEADROOM])
    style(fig, title, None, ylabel)
    if len(g) < 2:
        return fig, "Only one period is present in this selection."
    pct = pct_change(g[PERIOD_RECESSION], g[PERIOD_NON])
    fig.add_annotation(x=0.5, y=0.97, xref="paper", yref="paper", showarrow=False,
                       text=f"Recession vs. Non-Recession: {pct:+.1f}%", bgcolor="#F4F6F7", bordercolor="#D5D8DC")
    direction = "lower" if pct < 0 else "higher"
    return fig, (f"{what} is {abs(pct):.1f}% {direction} in recessions "
                 f"({g[PERIOD_RECESSION]:,.1f} vs {g[PERIOD_NON]:,.1f}).")


def chart_avg_sales_by_period(df: pd.DataFrame) -> Chart:
    """4. Average sales per record: Recession vs. Non-Recession."""
    return _period_bar(df, "Automobile_Sales", "Average automobile sales: Recession vs. Non-Recession",
                       "Average sales per record", "Average sales per record")


def chart_ad_by_period(df: pd.DataFrame) -> Chart:
    """9. Average advertising expenditure per record: Recession vs. Non-Recession."""
    return _period_bar(df, "Advertising_Expenditure", "Advertising expenditure: Recession vs. Non-Recession",
                       "Average advertising expenditure per record", "Average ad expenditure per record")


def chart_vehicle_by_period(df: pd.DataFrame) -> Chart:
    """5. Average sales per record by vehicle type and period (grouped bar)."""
    if df.empty:
        return empty_chart()
    g = df.groupby(["Vehicle_Label", "Period"])["Automobile_Sales"].mean().reset_index()
    order = df.groupby("Vehicle_Label")["Automobile_Sales"].mean().sort_values(ascending=False).index.tolist()
    fig = px.bar(g, x="Vehicle_Label", y="Automobile_Sales", color="Period", barmode="group",
                 color_discrete_map=PERIOD_COLORS, text_auto=".0f",
                 category_orders={"Vehicle_Label": order, "Period": PERIOD_ORDER})
    style(fig, "Vehicle type sales: Recession vs. Non-Recession", "Vehicle type", "Average sales per record")
    pivot = g.pivot(index="Vehicle_Label", columns="Period", values="Automobile_Sales")
    if {PERIOD_RECESSION, PERIOD_NON} <= set(pivot.columns):
        chg = ((pivot[PERIOD_RECESSION] - pivot[PERIOD_NON]) / pivot[PERIOD_NON] * 100).dropna()
        if not chg.empty:
            return fig, (f"Largest recession change: {chg.idxmin()} ({chg.min():+.1f}%); "
                         f"most resilient: {chg.idxmax()} ({chg.max():+.1f}%).")
    return fig, "Only one period is present for the selected vehicle types."


def chart_gdp(df: pd.DataFrame) -> Chart:
    """6. GDP distribution (box) and trend, built from the monthly frame."""
    m = build_monthly(df).dropna(subset=["GDP"]) if not df.empty else pd.DataFrame()
    if m.empty:
        return empty_chart()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65],
                        subplot_titles=("GDP distribution", "GDP over time"))
    for p in PERIOD_ORDER:
        sub = m[m["Period"] == p]
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub["GDP"], name=p, marker_color=PERIOD_COLORS[p], boxmean=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["GDP"], mode="markers", name=p, showlegend=False,
                                 marker=dict(color=PERIOD_COLORS[p], size=5),
                                 hovertemplate="%{x|%b %Y}<br>GDP: %{y:,.2f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=m["Date"], y=m["GDP"], mode="lines", showlegend=False,
                             line=dict(color="#95A5A6", width=1.2), hoverinfo="skip"), row=1, col=2)
    style(fig, "GDP variation: Recession vs. Non-Recession")
    fig.update_yaxes(title_text="GDP", row=1, col=1)
    med = m.groupby("Period")["GDP"].median()
    parts = [f"{p} {med[p]:,.2f}" for p in PERIOD_ORDER if p in med.index]
    return fig, "Median monthly GDP: " + " vs ".join(parts) + "."


def chart_seasonality(df: pd.DataFrame) -> Chart:
    """7. Seasonality: average sales per record by calendar month with Seasonality_Weight overlaid."""
    if df.empty:
        return empty_chart()
    g = df.groupby("Month_Num").agg(Avg_Sales=("Automobile_Sales", "mean"),
                                    Avg_Weight=("Seasonality_Weight", "mean")).reindex(range(1, 13))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=MONTHS, y=g["Avg_Sales"], name="Average sales per record",
                         marker_color=COLOR_PRIMARY, opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=MONTHS, y=g["Avg_Weight"], name="Average Seasonality_Weight", mode="lines+markers",
                             line=dict(color=COLOR_ACCENT, width=3)), secondary_y=True)
    style(fig, "Seasonality impact on automobile sales", "Month")
    fig.update_yaxes(title_text="Average sales per record", secondary_y=False)
    fig.update_yaxes(title_text="Average Seasonality_Weight", secondary_y=True, showgrid=False)
    if not g["Avg_Sales"].notna().any():
        return fig, NO_DATA_MSG
    r, n = pearson(g["Avg_Sales"], g["Avg_Weight"])
    link = f"; correlation with Seasonality_Weight across months r = {r:.2f}" if not np.isnan(r) else ""
    return fig, (f"Strongest month: {MONTHS[int(g['Avg_Sales'].idxmax()) - 1]}, "
                 f"weakest: {MONTHS[int(g['Avg_Sales'].idxmin()) - 1]}{link}.")


def chart_confidence_vs_sales(df: pd.DataFrame) -> Chart:
    """8. Consumer confidence vs. sales during recession (monthly aggregates, OLS)."""
    rec = df[df["Recession"] == 1] if not df.empty else df
    if rec.empty:
        return empty_chart("No recession months in this selection")
    m = build_monthly(rec).dropna(subset=["Consumer_Confidence", "Avg_Sales"])
    if m.empty:
        return empty_chart()
    fig = _scatter_ols(m, "Consumer_Confidence", "Avg_Sales", opacity=0.75,
                       color_discrete_sequence=[PERIOD_COLORS[PERIOD_RECESSION]], hover_data={"Date": "|%b %Y"})
    style(fig, "Consumer confidence vs. automobile sales (recession months)",
          "Consumer confidence", "Average sales per record")
    r, n = pearson(m["Consumer_Confidence"], m["Avg_Sales"])
    txt = f"r = {r:.2f}, n = {n}" if not np.isnan(r) else f"r n/a, n = {n}"
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper", showarrow=False, text=txt,
                       bgcolor="#F4F6F7", bordercolor="#D5D8DC")
    return fig, f"Monthly correlation during recessions: {txt}."


def chart_ad_by_vehicle(df: pd.DataFrame, vcolors: dict, repeat_share: float) -> Chart:
    """10. Total advertising expenditure per vehicle type during recession (donut)."""
    rec = df[df["Recession"] == 1] if not df.empty else df
    g = rec.groupby("Vehicle_Label")["Advertising_Expenditure"].sum().sort_values(ascending=False) if not rec.empty else pd.Series(dtype=float)
    g = g[g > 0]
    if g.empty:
        return empty_chart("No recession advertising data in this selection")
    fig = go.Figure(go.Pie(labels=list(g.index), values=g.values, hole=0.5, sort=False,
                           marker=dict(colors=[vcolors.get(k) for k in g.index]), textinfo="label+percent",
                           hovertemplate="%{label}<br>Total: %{value:,.0f}<br>%{percent}<extra></extra>"))
    style(fig, "Total advertising expenditure by vehicle type (recession)")
    note = f"{g.index[0]} takes {g.iloc[0] / g.sum():.0%} of recession ad spend."
    if repeat_share > 0:
        note += f" Caution: {repeat_share:.0%} of rows share an ad value, so totals may double count."
    return fig, note


def chart_unemployment_effect(df: pd.DataFrame, vcolors: dict) -> Chart:
    """11. Unemployment rate vs. sales by vehicle type during recession (bubble)."""
    rec = df[df["Recession"] == 1] if not df.empty else df
    if rec.empty:
        return empty_chart("No recession months in this selection")
    a = (rec.groupby(["Date", "Vehicle_Label"])
         .agg(Unemployment=("Unemployment_Rate", "mean"), Sales=("Automobile_Sales", "mean"), Price=("Price", "mean"))
         .reset_index().dropna(subset=["Unemployment", "Sales"]))
    if a.empty:
        return empty_chart()
    a["Price"] = a["Price"].fillna(a["Price"].median()) if a["Price"].notna().any() else 1.0
    fig = px.scatter(a, x="Unemployment", y="Sales", color="Vehicle_Label", size="Price", size_max=MAX_BUBBLE,
                     opacity=0.7, color_discrete_map=vcolors,
                     hover_data={"Date": "|%b %Y", "Price": ":,.0f", "Unemployment": ":.1f", "Sales": ":,.1f"})
    style(fig, "Unemployment rate vs. sales by vehicle type (recession, bubble = price)",
          "Unemployment rate (%)", "Average sales per record")
    lines = []
    for veh, sub in a.groupby("Vehicle_Label"):
        r, _ = pearson(sub["Unemployment"], sub["Sales"])
        lines.append(f"{veh}: " + (f"{r:+.2f}" if not np.isnan(r) else "n/a"))
    r_all, n_all = pearson(*[build_monthly(rec)[c] for c in ("Unemployment_Rate", "Avg_Sales")])
    overall = f"all vehicles r = {r_all:+.2f} (n = {n_all} months)" if not np.isnan(r_all) else "all vehicles r n/a"
    fig.add_annotation(x=0.99, y=0.99, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       align="left", text="<b>Correlation (r)</b><br>" + "<br>".join(lines),
                       bgcolor="#F4F6F7", bordercolor="#D5D8DC", font=dict(size=11))
    return fig, f"Correlation of unemployment with sales: {overall}."


def chart_state_map(df: pd.DataFrame) -> Chart:
    """12a. Choropleth of total recession-period sales by state."""
    rec = df[(df["Recession"] == 1)] if not df.empty else df
    sub = rec.dropna(subset=["State_Code"]) if not rec.empty else rec
    if sub.empty:
        return empty_chart("No mappable recession sales in this selection")
    agg = sub.groupby("State_Code").agg(State=("City", "first"), Total_Sales=("Automobile_Sales", "sum"),
                                        Avg_Ad=("Advertising_Expenditure", "mean"),
                                        Records=("Automobile_Sales", "size")).reset_index()
    top = (sub.groupby(["State_Code", "Vehicle_Label"])["Automobile_Sales"].sum().reset_index()
           .sort_values("Automobile_Sales", ascending=False).drop_duplicates("State_Code")
           .set_index("State_Code")["Vehicle_Label"])
    agg["Top_Vehicle"] = agg["State_Code"].map(top)
    fig = px.choropleth(agg, locations="State_Code", locationmode="USA-states", scope="usa", color="Total_Sales",
                        color_continuous_scale="Reds", hover_name="State",
                        hover_data={"State_Code": False, "Total_Sales": ":,.0f", "Top_Vehicle": True,
                                    "Avg_Ad": ":,.1f", "Records": True},
                        labels={"Total_Sales": "Total sales", "Top_Vehicle": "Top vehicle type",
                                "Avg_Ad": "Avg ad spend", "Records": "Records"})
    style(fig, "Highest-selling states during recession", height=MAP_HEIGHT)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), coloraxis_colorbar=dict(title="Total sales"))
    best = agg.sort_values("Total_Sales", ascending=False).iloc[0]
    return fig, f"{best['State']} leads recession-period sales with {best['Total_Sales']:,.0f} units ({best['Top_Vehicle']} is its top type)."


def chart_top_states(df: pd.DataFrame) -> Chart:
    """12b. Top-N states by recession-period sales."""
    rec = df[df["Recession"] == 1] if not df.empty else df
    if rec.empty:
        return empty_chart("No recession months in this selection")
    g = rec.groupby("City")["Automobile_Sales"].sum().sort_values(ascending=False).head(TOP_N_STATES).iloc[::-1]
    fig = go.Figure(go.Bar(x=g.values, y=list(g.index), orientation="h",
                           marker_color=PERIOD_COLORS[PERIOD_RECESSION], text=[f"{v:,.0f}" for v in g.values],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Total sales: %{x:,.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, float(g.max()) * Y_HEADROOM])
    style(fig, f"Top {TOP_N_STATES} states/regions (recession)", "Total sales", None, height=MAP_HEIGHT)
    return fig, f"Top {len(g)} states account for {g.sum():,.0f} units during recessions."


# --- Additional analyses (justification in each docstring) -------------------
def chart_correlation_heatmap(df: pd.DataFrame) -> Chart:
    """13a. Correlation heatmap on the monthly frame.

    Justification: a one-glance map of linear relationships, computed at monthly level so
    repeated month-level macro values are not counted several times.
    """
    labels = {"Avg_Sales": "Avg sales", "Avg_Price": "Avg price", "Avg_Ad": "Avg ad spend",
              "Avg_Seasonality": "Seasonality wt", "Competition": "Competition",
              "Consumer_Confidence": "Consumer conf.", "GDP": "GDP", "Growth_Rate": "Growth rate",
              "Unemployment_Rate": "Unemployment"}
    m = build_monthly(df) if not df.empty else pd.DataFrame()
    if len(m) < MIN_POINTS_CORR:
        return empty_chart()
    corr = m[list(labels)].corr().rename(index=labels, columns=labels)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    style(fig, "Correlation heatmap (monthly aggregates)")
    s = corr["Avg sales"].drop("Avg sales").dropna()
    caption = (f"Strongest linear link with sales: {s.abs().idxmax()} (r = {s[s.abs().idxmax()]:+.2f})."
               if not s.empty else "Correlations with sales are undefined for this selection.")
    return fig, caption


def chart_competition(df: pd.DataFrame) -> Chart:
    """13b. Sales by competition level.

    Justification: Competition is a market-pressure driver that the requested charts omit;
    a box plot of monthly average sales by level shows both level and spread.
    """
    m = build_monthly(df).dropna(subset=["Competition", "Avg_Sales"]) if not df.empty else pd.DataFrame()
    if m.empty:
        return empty_chart()
    m["Competition"] = m["Competition"].round().astype(int)
    fig = px.box(m, x="Competition", y="Avg_Sales", color="Period", color_discrete_map=PERIOD_COLORS,
                 category_orders={"Period": PERIOD_ORDER, "Competition": sorted(m["Competition"].unique())})
    fig.update_xaxes(type="category")
    style(fig, "Sales by competition level (monthly)", "Competition level", "Average sales per record")
    r, n = pearson(m["Competition"], m["Avg_Sales"])
    return fig, (f"Competition vs. sales: r = {r:+.2f} (n = {n})." if not np.isnan(r) else "Correlation undefined.")


def chart_growth_vs_sales(df: pd.DataFrame) -> Chart:
    """13c. Economic growth rate vs. sales.

    Justification: growth is the most direct cyclical indicator not yet shown; comparing its
    slope across regimes shows whether sales react differently to growth in recessions.
    """
    m = build_monthly(df).dropna(subset=["Growth_Rate", "Avg_Sales"]) if not df.empty else pd.DataFrame()
    if m.empty:
        return empty_chart()
    fig = _scatter_ols(m, "Growth_Rate", "Avg_Sales", color="Period", color_discrete_map=PERIOD_COLORS,
                       category_orders={"Period": PERIOD_ORDER}, opacity=0.7, hover_data={"Date": "|%b %Y"})
    style(fig, "Growth rate vs. automobile sales (monthly)", "Growth rate", "Average sales per record")
    return fig, _corr_by_period(m, "Growth_Rate", "Avg_Sales")


# =============================================================================
# Layout
# =============================================================================
@dataclass(frozen=True)
class Ctx:
    """Everything a tab needs to build its charts."""

    base: pd.DataFrame      # Year / Vehicle / State filters only (both periods)
    dfp: pd.DataFrame       # base additionally filtered by the Period selector
    metric: str
    window: int
    vcolors: dict
    ad_repeat_share: float


def graph_card(chart: Chart, width: int = 12) -> dbc.Col:
    """A card with a graph and its insight caption."""
    fig, caption = chart
    return dbc.Col(dbc.Card(dbc.CardBody([
        dcc.Graph(figure=fig, config=GRAPH_CONFIG),
        html.P(["Insight: ", caption], className="small text-muted mb-0 mt-1"),
    ]), className="shadow-sm"), md=width, className="mb-3")


def tab_overview(c: Ctx) -> list:
    return [dbc.Row([graph_card(chart_sales_over_time(c.dfp, c.metric, c.window))]),
            dbc.Row([graph_card(chart_seasonality(c.dfp))])]


def tab_recession(c: Ctx) -> list:
    return [dbc.Row([graph_card(chart_recession_windows(c.base, c.metric))]),
            dbc.Row([graph_card(chart_avg_sales_by_period(c.base), 6),
                     graph_card(chart_vehicle_by_period(c.base), 6)])]


def tab_marketing(c: Ctx) -> list:
    return [dbc.Row([graph_card(chart_ad_vs_sales(c.base), 6), graph_card(chart_ad_by_period(c.base), 6)]),
            dbc.Row([graph_card(chart_ad_by_vehicle(c.base, c.vcolors, c.ad_repeat_share))])]


def tab_economic(c: Ctx) -> list:
    return [dbc.Row([graph_card(chart_gdp(c.base), 6), graph_card(chart_confidence_vs_sales(c.base), 6)]),
            dbc.Row([graph_card(chart_unemployment_effect(c.base, c.vcolors))]),
            dbc.Row([graph_card(chart_correlation_heatmap(c.base), 6), graph_card(chart_growth_vs_sales(c.base), 6)]),
            dbc.Row([graph_card(chart_competition(c.base))])]


def tab_geography(c: Ctx) -> list:
    return [dbc.Row([graph_card(chart_state_map(c.base), 8), graph_card(chart_top_states(c.base), 4)])]


TAB_BUILDERS = {"tab-overview": tab_overview, "tab-recession": tab_recession, "tab-marketing": tab_marketing,
                "tab-economic": tab_economic, "tab-geography": tab_geography}
TAB_LABELS = [("tab-overview", "Overview"), ("tab-recession", "Recession Analysis"), ("tab-marketing", "Marketing"),
              ("tab-economic", "Economic Factors"), ("tab-geography", "Geography")]


def kpi_card(label: str, value: str, sub: str = "", color: str = None) -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="text-muted small text-uppercase"),
        html.H3(value, className="mb-0", style={"color": color} if color else {}),
        html.Div(sub, className="text-muted small"),
    ]), className="shadow-sm h-100"), md=3, className="mb-3")


def build_kpis(base: pd.DataFrame, dfp: pd.DataFrame) -> list:
    """Four KPI cards: total sales, average per record, recession delta, average ad spend."""
    if dfp.empty:
        return [kpi_card(t, "-") for t in ("Total sales", "Average sales per record", "Recession vs. non-recession", "Average ad spend")]
    g = base.groupby("Period")["Automobile_Sales"].mean()
    delta = pct_change(g.get(PERIOD_RECESSION), g.get(PERIOD_NON))
    delta_txt = "n/a" if np.isnan(delta) else f"{delta:+.1f}%"
    delta_color = None if np.isnan(delta) else ("#C0392B" if delta < 0 else "#1E8449")
    return [
        kpi_card("Total sales", f"{dfp['Automobile_Sales'].sum():,.0f}", "units, current Period filter"),
        kpi_card("Average sales per record", f"{dfp['Automobile_Sales'].mean():,.1f}", "units per Date x State x Vehicle"),
        kpi_card("Recession vs. non-recession", delta_txt, "change in average sales per record", delta_color),
        kpi_card("Average ad spend", f"{dfp['Advertising_Expenditure'].mean():,.1f}", "per record (units as supplied)"),
    ]


def _field(label: str, control) -> html.Div:
    return html.Div([html.Label(label, className="small fw-semibold mb-1"), control], className="mb-3")


def build_layout(df: pd.DataFrame, report: DataReport) -> dbc.Container:
    """Header, filter sidebar, KPI row and tabbed chart area."""
    years = [int(y) for y in sorted(df["Year"].unique())]
    vehicles = sorted(df["Vehicle_Label"].unique())
    states = sorted(df["City"].unique())
    sidebar = dbc.Card(dbc.CardBody([
        html.H6("Filters", className="text-uppercase text-muted"),
        _field("Year", dcc.Dropdown(id="f-year", options=[{"label": str(y), "value": y} for y in years],
                                    value=years, multi=True)),
        _field("Vehicle type", dcc.Dropdown(id="f-vehicle", options=vehicles, value=vehicles, multi=True)),
        _field("State / Region", dcc.Dropdown(id="f-state", options=states, value=states, multi=True)),
        _field("Period", dcc.Dropdown(id="f-period", options=[PERIOD_ALL] + PERIOD_ORDER, value=PERIOD_ALL, clearable=False)),
        _field("Sales metric (time-series charts)", dcc.Dropdown(
            id="f-metric", options=[{"label": "Average per record", "value": METRIC_AVG},
                                    {"label": "Total", "value": METRIC_TOTAL}],
            value=METRIC_AVG, clearable=False)),
        _field("Rolling average (Overview)", dcc.Dropdown(
            id="f-roll", options=[{"label": lbl, "value": v} for v, lbl in ROLLING_OPTIONS], value=0, clearable=False)),
        html.P("Period applies to Overview and the KPI cards. Comparison charts always show both periods; "
               "recession-specific charts always show recessions.", className="small text-muted mb-0"),
    ]), className="shadow-sm", style={"position": "sticky", "top": "12px"})

    notes = []
    if report.warnings:
        notes.append(dbc.Alert([html.Strong("Data-quality notes"), html.Ul([html.Li(w) for w in report.warnings], className="mb-0")],
                               color="warning", dismissable=True, className="small"))
    return dbc.Container([
        html.Div([html.H3("Automobile Sales Dashboard", className="mb-0"),
                  html.Div("Recession vs. non-recession performance", className="small opacity-75")],
                 className="bg-primary text-white p-3 mb-3 rounded"),
        *notes,
        dbc.Row([
            dbc.Col(sidebar, md=3, className="mb-3"),
            dbc.Col([dbc.Row(id="kpi-row"),
                     dbc.Tabs([dbc.Tab(label=lbl, tab_id=tid) for tid, lbl in TAB_LABELS], id="tabs",
                              active_tab="tab-overview", className="mb-3"),
                     html.Div(id="tab-content")], md=9),
        ]),
    ], fluid=True, className="py-3")


# =============================================================================
# Callbacks and app factory
# =============================================================================
def register_callbacks(app: Dash, df: pd.DataFrame, report: DataReport) -> None:
    """One callback: filters + active tab -> KPI row and the active tab's charts."""
    vehicles = sorted(df["Vehicle_Label"].unique())
    vcolors = {v: VEHICLE_PALETTE[i % len(VEHICLE_PALETTE)] for i, v in enumerate(vehicles)}

    @app.callback(
        Output("kpi-row", "children"), Output("tab-content", "children"),
        Input("tabs", "active_tab"), Input("f-year", "value"), Input("f-vehicle", "value"),
        Input("f-state", "value"), Input("f-period", "value"), Input("f-metric", "value"), Input("f-roll", "value"),
    )
    def update(tab, years, vehs, states, period, metric, roll):
        base = apply_filters(df, years, vehs, states)
        dfp = base if period in (None, PERIOD_ALL) else base[base["Period"] == period]
        ctx = Ctx(base=base, dfp=dfp, metric=metric or METRIC_AVG, window=int(roll or 0),
                  vcolors=vcolors, ad_repeat_share=report.ad_repeat_share)
        content = TAB_BUILDERS.get(tab, tab_overview)(ctx)
        if base.empty:
            content.insert(0, dbc.Alert(f"{NO_DATA_MSG}. Adjust the filters.", color="info"))
        return build_kpis(base, dfp), content


def build_app(df: pd.DataFrame, report: DataReport) -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Automobile Sales Dashboard")
    app.layout = build_layout(df, report)
    register_callbacks(app, df, report)
    return app


def main() -> None:
    try:
        raw = load_data(DATA_PATH)
    except Exception as exc:
        log.error("Could not load data from %r: %s", DATA_PATH, exc)
        sys.exit(1)
    df, report = prepare_data(raw)
    log.info("Loaded %d rows, %s to %s", len(df), df["Date"].min().date(), df["Date"].max().date())
    build_app(df, report).run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
