import streamlit as st

st.set_page_config(page_title="Causal Stock Analyzer", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

from causal_analysis import compute_returns, granger_causes_all, pcmci_causes_fast, plot_causal_graph_lags
from forecasting import prepare_lstm_data, train_evaluate_lstm

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("stock_data_with_spy.csv", index_col="Date", parse_dates=True)
        if df.empty:
            st.error("Data file is empty.")
            return None
        return df
    except FileNotFoundError:
        st.error("File stock_data_with_spy.csv not found. Make sure it is in the same folder as app.py.")
        return None

def main():
    st.title("📈 Causal Stock Analyzer")
    st.markdown("Choose method: **Granger** (classic) or **PCMCI** (modern).")

    df_prices = load_data()
    if df_prices is None:
        return

    st.sidebar.header("🎯 Analysis Parameters")
    all_tickers = df_prices.columns.tolist()
    target = st.sidebar.selectbox("Target company", all_tickers, index=0)
    causal_method = st.sidebar.radio("Causal discovery method", ["Granger", "PCMCI"], index=0)

    max_lag = st.sidebar.slider("Maximum lag (days)", 1, 5, 3)
    alpha = st.sidebar.select_slider("Significance level (alpha)", options=[0.01, 0.05, 0.1], value=0.05)

    if causal_method == "Granger":
        fdr_control = st.sidebar.checkbox("Control False Discovery Rate (FDR)", value=False)
        top_n = None
    else:
        top_n = st.sidebar.slider("Number of companies to analyze (PCMCI)", 30, 200, 100, step=10)
        fdr_control = False

    st.sidebar.header("🧠 LSTM Parameters")
    seq_len = st.sidebar.slider("Sequence length (LSTM window)", 10, 60, 30)
    epochs = st.sidebar.number_input("LSTM training epochs", 10, 100, 20, step=10)
    lr = st.sidebar.number_input("Learning rate", 0.0001, 0.01, 0.0005, step=0.0005, format="%.4f")
    hidden_dim = st.sidebar.number_input("Hidden layer size", 32, 128, 64, step=32)
    use_spy = st.sidebar.checkbox("Add SPY (macro factor)", value=True)
    use_technical = st.sidebar.checkbox("Technical indicators (RSI, SMA14)", value=False)

    st.sidebar.header("🖼️ Graph Visualization")
    max_display = st.sidebar.slider("Number of links to display on graph", 10, 100, 50, step=10)

    if st.sidebar.button("🔍 Run analysis", type="primary"):
        st.session_state.clear()
        with st.spinner(f"Discovering causes using {causal_method}..."):
            returns = compute_returns(df_prices)
            if causal_method == "Granger":
                cause_lags = granger_causes_all(returns, target, max_lag=max_lag, alpha=alpha, fdr_control=fdr_control)
            else:
                try:
                    cause_lags = pcmci_causes_fast(df_prices, target, max_lag=max_lag, alpha=alpha, top_n=top_n)
                except Exception as e:
                    st.error(f"PCMCI error: {e}")
                    st.stop()
        if not cause_lags:
            st.warning("No significant causal links found. Try changing parameters.")
            return

        st.success(f"✅ Found {len(cause_lags)} causal links")
        st.dataframe(pd.DataFrame(cause_lags, columns=["Company", "Lag", "p-value"]), use_container_width=True)

        st.subheader("🔗 Causal graph")
        fig_graph = plot_causal_graph_lags(target, cause_lags, max_display=max_display)
        st.plotly_chart(fig_graph, use_container_width=True)

        st.session_state['cause_lags'] = cause_lags
        st.session_state['target'] = target
        st.session_state['causal_method'] = causal_method
        st.session_state['seq_len'] = seq_len
        st.session_state['epochs'] = epochs
        st.session_state['lr'] = lr
        st.session_state['hidden_dim'] = hidden_dim
        st.session_state['use_spy'] = use_spy
        st.session_state['use_technical'] = use_technical
        st.session_state['analysis_done'] = True

    if 'analysis_done' in st.session_state:
        st.subheader("🤖 LSTM training and comparison with random baseline")
        if st.button("Train model"):
            with st.spinner("Preparing data and training LSTM..."):
                target = st.session_state['target']
                cause_lags = st.session_state['cause_lags']
                causal_method = st.session_state['causal_method']
                seq_len = st.session_state['seq_len']
                epochs = st.session_state['epochs']
                lr = st.session_state['lr']
                hidden_dim = st.session_state['hidden_dim']
                use_spy = st.session_state['use_spy']
                use_technical = st.session_state['use_technical']
                macro = 'SPY' if (use_spy and 'SPY' in df_prices.columns) else None

                df_causal = df_prices.copy()
                causal_features = []
                for cause, lag, _ in cause_lags:
                    col_name = f"{cause}_lag{lag}"
                    df_causal[col_name] = df_prices[cause].shift(lag)
                    causal_features.append(col_name)
                if macro:
                    if macro not in causal_features:
                        df_causal[macro] = df_prices[macro]
                        causal_features.append(macro)
                causal_features = list(dict.fromkeys(causal_features))

                res = prepare_lstm_data(
                    df_causal, target, causal_features,
                    lookback=1, seq_len=seq_len,
                    macro_col=None, technical=use_technical
                )
                if len(res) != 8:
                    st.error("Error preparing data for causal model")
                    return
                X_train, X_test, y_train, y_test, _, scaler_y, _, current_price_test = res

                if len(X_train) < 10:
                    st.error("Not enough data for LSTM, reduce seq_len")
                    return

                all_except_target = [t for t in df_prices.columns if t != target]
                n_baseline = len(cause_lags)
                if n_baseline > len(all_except_target):
                    n_baseline = len(all_except_target)
                    base_comps = all_except_target.copy()
                else:
                    base_comps = random.sample(all_except_target, n_baseline)
                if macro:
                    if macro not in base_comps:
                        base_comps.append(macro)
                base_comps = list(dict.fromkeys(base_comps))

                res_b = prepare_lstm_data(
                    df_prices, target, base_comps,
                    lookback=1, seq_len=seq_len,
                    macro_col=None, technical=use_technical
                )
                if len(res_b) != 8:
                    st.error("Error preparing data for baseline")
                    return
                X_train_b, X_test_b, y_train_b, y_test_b, _, _, _, current_price_test_b = res_b

                _, y_true_c, y_pred_c, mae_c, rmse_c, r2_c, mape_c = train_evaluate_lstm(
                    X_train, y_train, X_test, y_test, scaler_y, current_price_test,
                    epochs=epochs, lr=lr, hidden_dim=hidden_dim
                )
                _, y_true_b, y_pred_b, mae_b, rmse_b, r2_b, mape_b = train_evaluate_lstm(
                    X_train_b, y_train_b, X_test_b, y_test_b, scaler_y, current_price_test_b,
                    epochs=epochs//2, lr=lr, hidden_dim=hidden_dim
                )

                st.subheader("📊 Model comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Causal model ({causal_method} + LSTM)**")
                    st.metric("MAE", f"{mae_c:.2f}")
                    st.metric("RMSE", f"{rmse_c:.2f}")
                    st.metric("R²", f"{r2_c:.3f}")
                    st.metric("MAPE", f"{mape_c:.1f}%")
                with col2:
                    st.markdown(f"**Baseline ({n_baseline} random companies + LSTM)**")
                    st.metric("MAE", f"{mae_b:.2f}")
                    st.metric("RMSE", f"{rmse_b:.2f}")
                    st.metric("R²", f"{r2_b:.3f}")
                    st.metric("MAPE", f"{mape_b:.1f}%")

                if mape_c < mape_b:
                    st.success("✅ Causal model performs better!")
                else:
                    st.warning("⚠️ Baseline performs better. Try adjusting parameters.")

                limit = min(200, len(y_true_c))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(limit), y=y_true_c[:limit], mode='lines', name='Actual', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=np.arange(limit), y=y_pred_c[:limit], mode='lines', name='Causal LSTM', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=np.arange(limit), y=y_pred_b[:limit], mode='lines', name='Baseline LSTM', line=dict(color='red', dash='dot')))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Top 10 causal links by significance")
                top_causes = sorted(cause_lags, key=lambda x: x[2])[:10]
                st.dataframe(pd.DataFrame(top_causes, columns=["Company", "Lag", "p-value"]), use_container_width=True)

if __name__ == "__main__":
    main()