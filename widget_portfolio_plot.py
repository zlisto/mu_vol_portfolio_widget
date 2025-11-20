# widget_portfolio_plot_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from trade_simulator import TradeSimulator
import plotly.graph_objects as go
import glob

st.set_page_config(page_title="Crypto Portfolio", layout="wide")

# ---- Global style tweaks (bigger fonts for mobile) ----
st.markdown(
    """
    <style>
    /* Make labels bigger and bolder */
    label, .stSelectbox label, .stDateInput label {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }

    /* Make select dropdown text bigger */
    .stSelectbox div[data-baseweb="select"] span {
        font-size: 1.3rem !important;
    }
    
    /* Make select dropdown options bigger */
    .stSelectbox div[data-baseweb="select"] {
        font-size: 1.3rem !important;
    }
    
    /* Make selectbox container bigger for touch */
    .stSelectbox > div {
        min-height: 48px !important;
    }

    /* Make date input text bigger */
    .stDateInput input {
        font-size: 1.3rem !important;
        padding: 12px !important;
        min-height: 48px !important;
    }
    
    /* Make date picker bigger */
    .stDateInput > div {
        font-size: 1.3rem !important;
    }

    /* Make buttons bigger and more touch-friendly */
    .stButton > button {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        padding: 16px 32px !important;
        min-height: 56px !important;
        width: 100% !important;
    }

    /* Make title bigger */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    /* Make subheaders bigger */
    h2, h3 {
        font-size: 2rem !important;
        font-weight: 600 !important;
    }

    /* Make body text bigger */
    .stMarkdown, .stText {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
    }

    /* Make captions bigger */
    .stCaption {
        font-size: 1.1rem !important;
    }

    /* Make table text bigger */
    .stDataFrame, .stTable, [data-testid="stDataFrame"], [data-testid="stTable"] {
        font-size: 36px !important;
    }
    
    .stDataFrame table, .stTable table, [data-testid="stDataFrame"] table, [data-testid="stTable"] table {
        font-size: 36px !important;
    }
    
    .stDataFrame th, .stTable th, [data-testid="stDataFrame"] th, [data-testid="stTable"] th,
    .stDataFrame thead th, .stTable thead th {
        font-size: 36px !important;
        font-weight: 600 !important;
        padding: 16px !important;
    }
    
    .stDataFrame td, .stTable td, [data-testid="stDataFrame"] td, [data-testid="stTable"] td,
    .stDataFrame tbody td, .stTable tbody td {
        font-size: 36px !important;
        padding: 16px !important;
    }
    
    /* Target all table elements more aggressively */
    table {
        font-size: 36px !important;
    }
    
    table th, table td {
        font-size: 36px !important;
    }

    /* Make error/warning messages bigger */
    .stAlert {
        font-size: 1.2rem !important;
    }
    
    .stAlert > div {
        font-size: 1.2rem !important;
    }

    /* Make plotly chart text bigger */
    .js-plotly-plot {
        font-size: 1.2rem !important;
    }

    /* Better spacing for mobile */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Crypto Portfolio")

# --- Controls ---
pairs_options = [5, 10, 20, 30, 40, 50]
npairs_portfolio = st.selectbox(
    "Number of pairs in portfolio",
    pairs_options,
    index=pairs_options.index(10),
)

start_date = st.date_input(
    "Start date",
    value=datetime(2020, 1, 1),
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2025, 10, 1),
)

period_days = 30  # fixed based on your filename pattern

# Build filename from selection (using daily returns files)
filename_pattern = f"data/returns_mu_vol_volume_{npairs_portfolio}_pairs_*_period_{period_days}_days_hodl_vs_signal_daily.csv"
matching_files = glob.glob(filename_pattern)

if matching_files:
    filename = matching_files[0]  # Use first match
else:
    # Fallback to standard pattern
    filename = f"data/returns_mu_vol_volume_{npairs_portfolio}_pairs_2020-01-01_period_{period_days}_days_hodl_vs_signal_daily.csv"

# --- Load data ---
try:
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
except FileNotFoundError:
    st.error(f"File not found: {filename}")
    st.stop()

df.fillna(0, inplace=True)

returns_hodl = df["HODL Portfolio"]
returns_signal = df["Signal Portfolio"]
returns_btc = df["BTC"]

mask = df.index >= pd.to_datetime(start_date)
df = df[mask]

if df.empty:
    st.warning("No data available for the selected start date.")
    st.stop()

returns_hodl = returns_hodl[mask]
returns_signal = returns_signal[mask]
returns_btc = returns_btc[mask]

# --- Compute cumulative returns & equity curves ---
cum_hodl = (1 + returns_hodl).cumprod() - 1
cum_signal = (1 + returns_signal).cumprod() - 1
cum_btc = (1 + returns_btc).cumprod() - 1

equity_hodl = (1 + cum_hodl).clip(lower=1e-8) * 100
equity_signal = (1 + cum_signal).clip(lower=1e-8) * 100
equity_btc = (1 + cum_btc).clip(lower=1e-8) * 100

# --- Metrics using your TradeSimulator ---
ts = TradeSimulator()
bars_per_year = 365  # daily returns (was 365 * 24 * 60 for minute-level)

metrics_btc = ts.compute_metrics(returns_btc, bars_per_year=bars_per_year)
metrics_signal = ts.compute_metrics(returns_signal, bars_per_year=bars_per_year)
metrics_hodl = ts.compute_metrics(returns_hodl, bars_per_year=bars_per_year)

# --- Display metrics nicely above plot ---
st.subheader("Performance Metrics")

metrics_df = pd.DataFrame(
    {
        "BTC": metrics_btc,
        "Signal": metrics_signal,
        "HODL": metrics_hodl,
    }
).T

# Create HTML table with large font
# Map column names to display names
col_display_names = {
    "annualized_returns": "Annualized Returns [%]",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown [%]"
}

html_table = "<table style='font-size: 36px !important; width: 100%; border-collapse: collapse;'>"
html_table += "<thead><tr>"
html_table += "<th style='font-size: 36px !important; padding: 16px; text-align: left; border: 1px solid #ddd;'>Strategy</th>"
for col in metrics_df.columns:
    display_name = col_display_names.get(col, col)
    html_table += f"<th style='font-size: 36px !important; padding: 16px; text-align: left; border: 1px solid #ddd;'>{display_name}</th>"
html_table += "</tr></thead><tbody>"
for idx, row in metrics_df.iterrows():
    html_table += "<tr>"
    html_table += f"<td style='font-size: 36px !important; padding: 16px; font-weight: 600; border: 1px solid #ddd;'>{idx}</td>"
    for col, val in row.items():
        # Format as percentage for annualized_returns and max_drawdown
        if col in ["annualized_returns", "max_drawdown"]:
            formatted_val = f"{val * 100:.2f}%"
        else:
            formatted_val = f"{val:.4f}"
        html_table += f"<td style='font-size: 36px !important; padding: 16px; border: 1px solid #ddd;'>{formatted_val}</td>"
    html_table += "</tr>"
html_table += "</tbody></table>"

st.markdown(html_table, unsafe_allow_html=True)

# --- Interactive Plotly chart (dark, same colors, clickable legend) ---
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=equity_signal,
        mode="lines",
        name="Signal Portfolio",
        line=dict(color="orange"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=equity_hodl,
        mode="lines",
        name="HODL Portfolio",
        line=dict(color="blue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=equity_btc,
        mode="lines",
        name="BTC",
        line=dict(color="green"),
    )
)

fig.update_layout(
        title=dict(
            text=f"{npairs_portfolio}-Pair Portfolio (rebalance every 30 days)",
            font=dict(size=24)
        ),
        xaxis_title=dict(text="Time", font=dict(size=20)),
        yaxis_title=dict(text="Cumulative Returns [%] (log scale)", font=dict(size=20)),
        xaxis=dict(
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            type="log",
            tickfont=dict(size=18)
        ),
        template="plotly_dark",  # dark background
        legend=dict(
            title_text="Series",
            title_font=dict(size=20),
            font=dict(size=18),
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        ),
        hovermode="x unified",
        font=dict(size=18),  # Default font size for all text
        height=700,  # Make plot taller
    )

st.plotly_chart(fig, use_container_width=True)
