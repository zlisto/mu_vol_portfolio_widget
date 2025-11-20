Write-Host "Starting Streamlit app with auto-reload enabled..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will automatically reload when you save changes to Python files." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host ""
streamlit run widget_portfolio_plot.py --server.runOnSave true

