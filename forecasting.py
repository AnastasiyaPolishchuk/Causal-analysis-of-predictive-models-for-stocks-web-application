import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def prepare_features_target(df, target_col, feature_cols, lookback=1, lags=0, macro_col=None, technical=False):
    X = df[feature_cols].copy()
    macro_present = False
    if macro_col and macro_col in df.columns:
        X[macro_col] = df[macro_col]
        macro_present = True
    if technical:
        tech_cols = feature_cols.copy()
        if macro_present:
            tech_cols.append(macro_col)
        for col in tech_cols:
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            X[f'{col}_RSI'] = 100 - (100 / (1 + rs))
            X[f'{col}_SMA14'] = df[col].rolling(window=14).mean()
    if lags > 0:
        X_orig = X.copy()
        for lag in range(1, lags+1):
            lagged = X_orig.shift(lag).add_suffix(f'_lag{lag}')
            X = pd.concat([X, lagged], axis=1)
    y = df[target_col].shift(-lookback)
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(columns=[target_col])
    y = combined[target_col]
    return X, y

def prepare_lstm_data(df, target_col, feature_cols, lookback=1, seq_len=20, macro_col=None, technical=False):
    X, y_price = prepare_features_target(df, target_col, feature_cols,
                                          lookback=lookback, lags=0,
                                          macro_col=macro_col, technical=technical)
    prices = df[target_col]
    common_idx = X.index.intersection(prices.index)
    X = X.loc[common_idx]
    y_price = y_price.loc[common_idx]
    current_price = prices.loc[common_idx]
    y = np.log(y_price / current_price)
    valid = ~(np.isnan(y) | np.isinf(y))
    X = X[valid]
    y = y[valid]
    current_price = current_price[valid]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len + 1):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len-1])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    current_price_train = current_price.iloc[seq_len-1:].values[:split]
    current_price_test = current_price.iloc[seq_len-1:].values[split:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, current_price_train, current_price_test

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

def train_evaluate_lstm(X_train, y_train, X_test, y_test, scaler_y, current_price_test, epochs=50, batch_size=32, lr=0.001, hidden_dim=64, num_layers=2, dropout=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_dim=X_train.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_t).cpu().numpy()

    y_pred_returns = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true_returns = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    base_prices = current_price_test[:len(y_true_returns)]
    y_pred_price = base_prices * np.exp(y_pred_returns[:len(base_prices)])
    y_true_price = base_prices * np.exp(y_true_returns[:len(base_prices)])

    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2 = r2_score(y_true_price, y_pred_price)
    mape = np.mean(np.abs((y_true_price - y_pred_price) / y_true_price)) * 100

    return model, y_true_price, y_pred_price, mae, rmse, r2, mape