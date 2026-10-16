Causal analysis of predictive models for stocks, web application.

The purpose of this project was to discover casual structures, validate their persistence and design and implement a web application that finds stable, explanatory relationships and apply the framework to S&P500 index.

Key Achievements:

1.	Causal Discovery Implementation
The Granger causality algorithm, offers a fast and interpretable way to identify potential predictive relationships. The PCMCI algorithm provides a more robust and time-series specific causal discovery method. Both algorithms return a ranked list of cause effect pairs with temporal lags, which are then used as feature sets for the forecasting model.
2.	Predictive Model LSTM
A Long Short Term Memory network was chosen for its ability to capture both short term fluctuations and long term dependencies in sequential data. The model was trained on lagged features derived from the discovered causal relationships, plus optional market index (SPY) and technical indicators. A baseline LSTM model was trained on an equal number of randomly selected companies to evaluate the added value of causal feature selection.
3.	Web Application
Built with Streamlit, the application consists of three modular components: app.py (user interface), causal_analysis.py (causal discovery), and forecasting.py (LSTM training and evaluation). Users can choose between Granger and PCMCI, adjust key parameters (e.g., max_lag, alpha, top_n), and visualise the resulting causal graph using interactive Plotly charts. After causal discovery, the application trains an LSTM model on the selected features and compares its performance (MAE, RMSE, R², MAPE) with the random baseline, clearly indicating whether causal discovery improves prediction.
4.	Experimental Results
The evaluation was performed on daily closing prices of S&P 500 stocks from 2017 to 2026. Using Apple Inc. (AAPL) as the target company, the PCMCI method discovered approximately 100–150 causal links with various time lags. The discovered causes align with market intuition – AAPL is strongly influenced by other technology giants (MSFT, GOOGL, AMZN) and by component suppliers (SWKS). Financial and industrial companies (JPM, ALLE, TEL) show negligible influence. The causal LSTM model achieved a MAE of about 2.69, while the random baseline LSTM scored 2.70. Although the difference is modest, it confirms that causal feature selection provides a slight but consistent benefit.
