# Crypto Portfolio Viewer

An interactive Streamlit web application for visualizing and comparing crypto portfolio performance metrics. Compare Signal Portfolio, HODL Portfolio, and BTC performance with interactive charts and detailed metrics.

## Features

- 📊 Interactive portfolio performance visualization
- 📈 Compare three strategies: Signal Portfolio, HODL Portfolio, and BTC
- 📱 Mobile-friendly interface with large fonts
- 📉 Performance metrics: Annualized Returns, Sharpe Ratio, Max Drawdown
- 🔄 Auto-updating charts when parameters change
- 📅 Customizable date range and portfolio size

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zlisto/mu_vol_portfolio_widget.git
cd mu_vol_portfolio_widget
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

Run the Streamlit app:

**Windows (PowerShell):**
```powershell
.\run_app.ps1
```

**Windows (Command Prompt):**
```cmd
run_app.bat
```

**Or directly:**
```bash
streamlit run widget_portfolio_plot.py
```

The app will automatically reload when you make changes to the code.

### Online Deployment

This app can be deployed to:
- [Streamlit Cloud](https://streamlit.io/cloud) - Free hosting for Streamlit apps
- [Heroku](https://www.heroku.com/)
- Any platform that supports Python/Streamlit

#### Deploying to Streamlit Cloud

1. Push your code to GitHub (already done!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository: `zlisto/mu_vol_portfolio_widget`
6. Set the main file path: `widget_portfolio_plot.py`
7. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## Data

The app uses daily returns data stored in the `data/` directory. Files are named:
- `returns_mu_vol_volume_{N}_pairs_*_period_30_days_hodl_vs_signal_daily.csv`

Where `{N}` is the number of pairs (5, 10, 20, 30, 40, or 50).

## Project Structure

```
mu_vol_portfolio_widget/
├── widget_portfolio_plot.py    # Main Streamlit app
├── trade_simulator.py           # TradeSimulator class for metrics
├── convert_to_daily.py          # Script to convert minute-level to daily data
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── data/
    └── *.csv                   # Daily returns data files
```

## Configuration

The app allows you to:
- Select number of pairs in portfolio (5, 10, 20, 30, 40, 50)
- Choose start date for analysis
- View performance metrics and interactive charts

## License

MIT License

## Author

zlisto

