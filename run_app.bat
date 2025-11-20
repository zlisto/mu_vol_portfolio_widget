@echo off
echo Starting Streamlit app with auto-reload enabled...
echo.
echo The app will automatically reload when you save changes to Python files.
echo Press Ctrl+C to stop the server.
echo.
streamlit run widget_portfolio_plot.py --server.runOnSave true

