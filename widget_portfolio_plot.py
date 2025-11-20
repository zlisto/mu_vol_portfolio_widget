# widget_portfolio_plot_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from trade_simulator import TradeSimulator
import plotly.graph_objects as go
import glob

st.set_page_config(page_title="Crypto Portfolio", layout="wide")

# --- Authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login")
    
    # Login form with larger fonts
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            font-size: 1.3rem !important;
            padding: 12px !important;
        }
        .stButton > button {
            font-size: 1.4rem !important;
            padding: 16px 32px !important;
            min-height: 56px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", key="username_input")
        password = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", use_container_width=True):
            if username == "zlisto" and password == "t1uh9dy1s5r":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.stop()

# Main app content (only shown if authenticated)

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
    
    /* Metrics table styling */
    .metrics-table {
        font-size: 36px !important;
    }
    
    .metrics-table th, .metrics-table td {
        font-size: 36px !important;
    }
    
    /* Mobile-specific styles */
    @media (max-width: 768px) {
        /* Smaller table font on mobile */
        table, .metrics-table {
            font-size: 16px !important;
        }
        
        table th, table td, .metrics-table th, .metrics-table td {
            font-size: 16px !important;
            padding: 8px 4px !important;
        }
        
        /* Make table scrollable horizontally on mobile */
        .stDataFrame, .stTable {
            overflow-x: auto !important;
            display: block !important;
        }
        
        /* Smaller title on mobile */
        h1 {
            font-size: 1.8rem !important;
        }
        
        h2, h3 {
            font-size: 1.5rem !important;
        }
        
        /* Reduce padding on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
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
except FileNotFoundError as e:
    st.error(f"File not found: {filename}")
    st.info("Please ensure the data files are available in the data/ directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

df.fillna(0, inplace=True)

# Check if required columns exist
required_columns = ["HODL Portfolio", "Signal Portfolio", "BTC"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns in data file: {', '.join(missing_columns)}")
    st.stop()

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
# Map column names to display names (with line breaks for two-row headers)
col_display_names = {
    "annualized_returns": "Ann.<br>Returns",
    "sharpe_ratio": "Sharpe<br>Ratio",
    "max_drawdown": "Max<br>Drawdown"
}

html_table = "<div style='overflow-x: auto;'>"
html_table += "<table class='metrics-table' style='width: 100%; border-collapse: collapse; min-width: 600px;'>"
html_table += "<thead><tr>"
html_table += "<th style='padding: 16px; text-align: left; border: 1px solid #ddd;'>Strategy</th>"
for col in metrics_df.columns:
    display_name = col_display_names.get(col, col)
    html_table += f"<th style='padding: 16px; text-align: left; border: 1px solid #ddd;'>{display_name}</th>"
html_table += "</tr></thead><tbody>"
for idx, row in metrics_df.iterrows():
    html_table += "<tr>"
    html_table += f"<td style='padding: 16px; font-weight: 600; border: 1px solid #ddd;'>{idx}</td>"
    for col, val in row.items():
        # Format as percentage for annualized_returns and max_drawdown
        if col in ["annualized_returns", "max_drawdown"]:
            formatted_val = f"{val * 100:.2f}%"
        elif col == "sharpe_ratio":
            formatted_val = f"{val:.2f}"
        else:
            formatted_val = f"{val:.4f}"
        html_table += f"<td style='padding: 16px; border: 1px solid #ddd;'>{formatted_val}</td>"
    html_table += "</tr>"
html_table += "</tbody></table></div>"

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
        yaxis_title=dict(text="Cumulative Returns [%]", font=dict(size=20)),
        xaxis=dict(
            tickfont=dict(size=18),
            automargin=True  # Auto-adjust margins to prevent squishing
        ),
        yaxis=dict(
            type="log",
            tickfont=dict(size=18),
            automargin=True  # Auto-adjust margins to prevent squishing
        ),
        template="plotly_dark",  # dark background
        legend=dict(
            title_text="Series",
            title_font=dict(size=20),
            font=dict(size=18),
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        ),
        hovermode="x unified",
        font=dict(size=18),  # Default font size for all text
        margin=dict(l=60, r=20, t=80, b=60),  # Better margins for mobile
        autosize=True,  # Make plot responsive
        height=500,  # Fixed height for consistent aspect ratio (good for mobile)
    )

# Use config for responsive display
config = {
    'displayModeBar': True,
    'responsive': True,
    'displaylogo': False,
}

st.plotly_chart(fig, use_container_width=True, config=config)
