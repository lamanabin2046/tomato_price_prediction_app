import os
import pandas as pd
import numpy as np
import joblib
from datetime import date
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Optional imports
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


# =========================================================
# ⚙️ Custom Lag & Rolling Configurations
# =========================================================
custom_lag_config = {
    "Average_Price":  [1, 3, 7],   # Short memory for main price trend
    "Sarlahi_Temperature": [1, 3, 7],
    "Sarlahi_Rainfall_MM": [1, 3, 7],
    "Hilly_Temperature": [1, 3, 7],
    "Hilly_Rainfall_MM": [1, 3, 7],
    "Kathmandu_Rainfall_MM": [1, 3,7],
    "Supply_Volume": [1, 3, 7],
}



custom_roll_config = {
    "Average_Price": [3, 7, 14],
    "Supply_Volume": [3, 7],
    "Sarlahi_Temperature": [3, 7],
    "Sarlahi_Rainfall_MM": [3, 7],
    "Hilly_Temperature": [3, 7],
    "Dhading_Rainfall_MM": [3, 7],
    "Kavre_Rainfall_MM": [3, 7],
    "Kathmandu_Rainfall_MM": [3, 7],
}


# =========================================================
# 📘 Load and Clean Dataset
# =========================================================
def load_data(path="data/processed/tomato_clean_data.csv"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    df = pd.read_csv(full_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    drop_cols = [c for c in df.columns if "FY" in c or "Fiscal" in c or "Year" in c]
    if drop_cols:
        print(f"🗑️ Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    print(f"✅ Loaded dataset ({len(df)} rows, {len(df.columns)} columns)")
    return df


# =========================================================
# 🧩 Add Custom Lag and Rolling Features
# =========================================================
def add_lag_and_rolling(df, lag_config=None, roll_config=None):
    """Add custom lag & rolling features per feature config."""
    df = df.copy()
    lag_config = lag_config or {}
    roll_config = roll_config or {}

    # --- Lags ---
    for col, lags in lag_config.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # --- Rolling windows ---
    for col, windows in roll_config.items():
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_roll{window}"] = df[col].rolling(window).mean()

    return df.dropna().reset_index(drop=True)


# =========================================================
# ⚙️ Cross-Validation with NaN-safe Pipeline
# =========================================================
def cross_validate_model(name, model, df, target, tscv):
    results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        print(f"\n🧩 Fold {fold} — Generating lag/rolling features...")

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Apply custom configs
        train_df = add_lag_and_rolling(train_df,
                                       lag_config=custom_lag_config,
                                       roll_config=custom_roll_config)
        val_df = pd.concat([train_df, val_df], ignore_index=True)
        val_df = add_lag_and_rolling(val_df,
                                     lag_config=custom_lag_config,
                                     roll_config=custom_roll_config).iloc[len(train_df):]

        X_train = train_df.drop(columns=[target, "Date"], errors="ignore")
        y_train = train_df[target]
        X_val = val_df.drop(columns=[target, "Date"], errors="ignore")
        y_val = val_df[target]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"✅ Fold {fold}: MAE={mae:.3f}, R²={r2:.3f}")

        results.append({"Fold": fold, "MAE": mae, "R2": r2})

    metrics = {
        "val_MAE": np.mean([r["MAE"] for r in results]),
        "val_MAE_std": np.std([r["MAE"] for r in results]),
        "val_R2": np.mean([r["R2"] for r in results]),
    }
    return metrics


# =========================================================
# 🧠 Train Models → Replace Old Models
# =========================================================
def train_all_models(df, target="Average_Price"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    results_dir = os.path.join(base_dir, "outputs/results")
    models_dir = os.path.join(base_dir, "outputs/models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    tscv = TimeSeriesSplit(n_splits=5)
    today = date.today().isoformat()

    models = {
        "linear_regression": LinearRegression(),
        "lasso_regression": Lasso(alpha=0.1, random_state=42),
        "ridge_regression": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boost": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    if XGBRegressor:
        print("✅ XGBoost detected.")
        models["xgboost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1
        )

    if LGBMRegressor:
        print("✅ LightGBM detected.")
        models["lightgbm"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    # Remove old models
    for f in os.listdir(models_dir):
        if f.endswith(".joblib"):
            os.remove(os.path.join(models_dir, f))
    print("🧹 Cleared old model files.")

    results = []
    for name, model in models.items():
        print(f"\n🔄 Training model: {name}")
        metrics = cross_validate_model(name, model, df, target, tscv)
        results.append({
            "Date": today,
            "Model": name,
            "Val_MAE": metrics["val_MAE"],
            "Val_MAE_std": metrics["val_MAE_std"],
            "Val_R2": metrics["val_R2"],
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, "cv_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📘 Saved updated CV results → {results_path}")

    # Determine Top 3
    top3 = results_df.sort_values(by="Val_MAE").head(3)
    top3_path = os.path.join(results_dir, "top3_models.csv")
    top3.to_csv(top3_path, index=False)
    print(f"🏆 Saved Top 3 Summary → {top3_path}")

    # Train and save only Top 3
    for _, row in top3.iterrows():
        model_name = row["Model"]
        print(f"\n💾 Final Training (Saving): {model_name}")

        model_class = models[model_name]
        df_full = add_lag_and_rolling(df.copy(),
                                      lag_config=custom_lag_config,
                                      roll_config=custom_roll_config)
        df_full = df_full.dropna()
        X = df_full.drop(columns=[target, "Date"], errors="ignore")
        y = df_full[target]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model_class)
        ])
        pipeline.fit(X, y)

        model_filename = f"{model_name}_latest.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(pipeline, model_path)
        joblib.dump(X.columns.tolist(), model_path.replace(".joblib", "_features.joblib"))
        print(f"✅ Saved Model → {model_path}")

    print("\n✅ Training complete! Old models replaced with current Top 3.")
    return results_df


# =========================================================
# 🚀 Main
# =========================================================
if __name__ == "__main__":
    print("📥 Loading dataset...")
    df = load_data()
    print("⚙️ Training and evaluating models with custom lag/roll config...")
    results = train_all_models(df)
    print("✅ All done — Only latest Top 3 models retained!")
