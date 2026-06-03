import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from statsmodels.tsa.stattools import grangercausalitytests
import io
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore")

def compute_returns(df):
    """Log returns."""
    return np.log(df / df.shift(1)).dropna()

def granger_causes_all(df_returns, target_col, max_lag=3, alpha=0.05, fdr_control=False):
    """
    Granger causality on all companies.
    Returns list of (cause, lag, p-value) for all significant links.
    """
    all_causes = []
    for col in df_returns.columns:
        if col == target_col:
            continue
        series_t = df_returns[target_col].dropna()
        series_c = df_returns[col].dropna()
        common_idx = series_t.index.intersection(series_c.index)
        series_t = series_t[common_idx]
        series_c = series_c[common_idx]
        best_pval = 1.0
        best_lag = 1
        for lag in range(1, max_lag+1):
            data = np.column_stack([series_t, series_c])
            with redirect_stdout(io.StringIO()):
                res = grangercausalitytests(data, maxlag=lag, verbose=False)
            p_val = res[lag][0]['ssr_ftest'][1]
            if p_val < best_pval:
                best_pval = p_val
                best_lag = lag
        if best_pval < alpha:
            all_causes.append((col, best_lag, best_pval))
    all_causes.sort(key=lambda x: x[2])
    return all_causes

try:
    from tigramite.independence_tests.parcorr import ParCorr
    print("ParCorr import successful (path 2)")
except ImportError:
    try:
        from tigramite.independence_tests import ParCorr
        print("ParCorr import successful (path 1)")
    except ImportError:
        ParCorr = None
        print("Could not import ParCorr. Please reinstall tigramite.")

def pcmci_causes_fast(df_prices, target_col, max_lag=3, alpha=0.05, top_n=100):
    """
    Fast PCMCI: analyzes only top-N correlated companies.
    """
    if ParCorr is None:
        raise ImportError("ParCorr not loaded. Check tigramite installation.")
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    
    returns = compute_returns(df_prices)
    corrs = returns.corrwith(returns[target_col]).abs().sort_values(ascending=False)
    candidates = corrs.drop(target_col, errors='ignore').head(top_n).index.tolist()
    subset_cols = [target_col] + candidates
    df_subset = returns[subset_cols].copy()
    data = df_subset.values
    data = data[~np.isnan(data).any(axis=1)]
    dataframe = pp.DataFrame(data)
    parcorr = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=alpha)
    
    if 'q_matrix' in results:
        p_matrix = results['q_matrix']
    elif 'p_matrix' in results:
        p_matrix = results['p_matrix']
    else:
        raise KeyError(f"Unknown PCMCI result format. Available keys: {list(results.keys())}")
    
    target_idx = 0
    cause_lags = []
    for j in range(1, len(subset_cols)):
        for tau in range(max_lag + 1):
            pval = p_matrix[j, target_idx, tau]
            if pval < alpha:
                cause_lags.append((subset_cols[j], tau, pval))
    cause_lags.sort(key=lambda x: x[2])
    return cause_lags

def plot_causal_graph_lags(target, cause_lags, max_display=50, title="Causal graph"):
    if not cause_lags:
        return go.Figure()
    display_lags = cause_lags[:max_display]
    G = nx.DiGraph()
    G.add_node(target, color='#d62728', size=30)
    for cause, lag, pval in display_lags:
        node_name = f"{cause}\n(lag {lag})"
        G.add_node(node_name, color='#aec7e8', size=20)
        G.add_edge(node_name, target, pval=pval)
    pos = nx.spring_layout(G, k=2, seed=42)
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        pval = edge[2]['pval']
        intensity = min(1.0, max(0.0, (-np.log10(pval) - 1) / 3))
        r = int(200 - 150 * intensity)
        g = int(80 + 100 * intensity)
        b = int(50 + 30 * intensity)
        color = f'rgba({r}, {g}, {b}, 0.8)'
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=2, color=color),
            hoverinfo='text',
            text=f"p = {pval:.4f}"
        ))
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[0])
        node_color.append(node[1]['color'])
        node_size.append(node[1]['size'])
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(size=node_size, color=node_color, line=dict(width=2)),
        hoverinfo='text'
    )
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title=title,
                                     showlegend=False, hovermode='closest',
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     width=900, height=700))
    return fig