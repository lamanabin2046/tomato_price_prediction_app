# utils/preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_internal_data(path="artifacts/data/clean_data_with_lag_roll.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Normalize date column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # convert booleans and TF-like text to 0/1
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    for col in df.select_dtypes(include=["object"]).columns:
        vals = set(df[col].dropna().astype(str).unique())
        if vals.issubset({"True","False","true","false","Yes","No","yes","no"}):
            df[col] = df[col].map({
                "True":1,"False":0,"true":1,"false":0,"Yes":1,"No":0,"yes":1,"no":0
            })
    # fill numeric NaNs
    for col in df.select_dtypes(include=["float","int"]).columns:
        df[col] = df[col].fillna(0)
    # ensure numeric-like strings convert
    df = df.apply(pd.to_numeric, errors="ignore")
    return df

def ensure_lags(df: pd.DataFrame, target_col="Average_Price", lags=(1,7,30), roll_windows=(7,30,90)):
    # Create lag and rolling mean features if missing
    for lag in lags:
        name = f"{target_col}_lag{lag}"
        if name not in df.columns:
            df[name] = df[target_col].shift(lag)
    for w in roll_windows:
        name = f"{target_col}_rollmean_{w}"
        if name not in df.columns:
            df[name] = df[target_col].rolling(window=w, min_periods=1).mean().shift(1)
    # If some NaNs introduced at top, fill by forward/backfill minimally
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def build_feature_matrix(df: pd.DataFrame, feature_cols=None):
    # Default features: many of the columns in your CSV useful for models
    if feature_cols is None:
        # exclude date & target
        exclude = {"date", "Date", "Average_Price", "avg_price", "Price"}
        feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    return X, feature_cols

def fit_and_save_scaler(X: pd.DataFrame, scaler_path="artifacts/scalers/feature_scaler.pkl"):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values)
    joblib.dump(scaler, scaler_path)
    return scaler, X_scaled

def load_scaler(scaler_path="artifacts/scalers/feature_scaler.pkl"):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)
    return joblib.load(scaler_path)

def create_sequences(X_scaled: np.ndarray, y: np.ndarray, seq_len: int = 30):
    """
    Build sequences for LSTM.
    X_scaled: (n_samples, n_features)
    y: (n_samples,)
    returns sequences X_seq: (n_seq, seq_len, n_features), y_seq: (n_seq,)
    where each y_seq corresponds to the label immediately after the sequence.
    """
    Xs, ys = [], []
    n = X_scaled.shape[0]
    for i in range(seq_len, n):
        Xs.append(X_scaled[i-seq_len:i, :])
        ys.append(y[i])
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys
