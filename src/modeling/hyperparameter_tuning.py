import os
import pandas as pd
import numpy as np
import joblib
from datetime import date
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import itertools

# Optional imports
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# =========================================================
# 📁 Base Directories
# =========================================================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
results_dir = os.path.join(base_dir, "outputs/results")
models_dir = os.path.join(base_dir, "outputs/models/tuned_models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)






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
# 🧩 Feature Engineering: Lags & Rolling Means
# =========================================================
def add_lag_and_rolling(df, lag_config=None, roll_config=None):
    """
    Add lag and rolling mean features to the dataframe based on configuration.
    """
    df = df.copy()
    lag_config = lag_config or {}
    roll_config = roll_config or {}

    # --- LAG FEATURES ---
    for col, lags in lag_config.items():
        if col not in df.columns:
            print(f"⚠️ Skipping lag for {col} (not in dataset)")
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # --- ROLLING FEATURES ---
    for col, windows in roll_config.items():
        if col not in df.columns:
            print(f"⚠️ Skipping roll for {col} (not in dataset)")
            continue
        for window in windows:
            df[f"{col}_roll{window}"] = df[col].rolling(window).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# =========================================================
# 📘 Load Dataset
# =========================================================
def load_data(path="data/processed/tomato_clean_data.csv"):
    full_path = os.path.join(base_dir, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    df = pd.read_csv(full_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Fiscal_Year" in df.columns:
        df = df.drop(columns=["Fiscal_Year"])
        print("🗑️ Dropped Fiscal_Year column")

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    print(f"✅ Loaded dataset → {len(df)} rows × {len(df.columns)} columns")
    return df


# =========================================================
# ⚙️ TimeSeries-Aware Model Tuning (NaN-safe)
# =========================================================
def tune_model(model_class, param_grid, X_full, y_full, alpha=0.5):
    """
    TimeSeries-aware tuning with NaN-safe pipeline.
    Objective = val_MAE + alpha * |train_MAE - val_MAE|
    """
    tscv = TimeSeriesSplit(n_splits=5)
    best_score = np.inf
    best_params = None
    best_results = {}

    keys, values = zip(*param_grid.items())

    for param_set in itertools.product(*values):
        params = dict(zip(keys, param_set))
        model = model_class(**params)

        train_mae_list, val_mae_list = [], []

        for train_idx, val_idx in tscv.split(X_full):
            X_train_raw, X_val_raw = X_full.iloc[train_idx].copy(), X_full.iloc[val_idx].copy()
            y_train_raw, y_val_raw = y_full.iloc[train_idx].copy(), y_full.iloc[val_idx].copy()

            train_df = X_train_raw.copy()
            train_df["Average_Price"] = y_train_raw
            val_df = X_val_raw.copy()
            val_df["Average_Price"] = y_val_raw

            # Add lag & rolling features (custom config)
            train_df = add_lag_and_rolling(train_df,
                                           lag_config=custom_lag_config,
                                           roll_config=custom_roll_config)
            val_df = pd.concat([train_df, val_df], ignore_index=True)
            val_df = add_lag_and_rolling(val_df,
                                         lag_config=custom_lag_config,
                                         roll_config=custom_roll_config).iloc[len(train_df):]

            X_train = train_df.drop(columns=["Average_Price"], errors="ignore")
            y_train = train_df["Average_Price"]
            X_val = val_df.drop(columns=["Average_Price"], errors="ignore")
            y_val = val_df["Average_Price"]

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            y_train_pred = pipeline.predict(X_train)
            y_val_pred = pipeline.predict(X_val)

            train_mae_list.append(mean_absolute_error(y_train, y_train_pred))
            val_mae_list.append(mean_absolute_error(y_val, y_val_pred))

        train_mae = np.mean(train_mae_list)
        val_mae = np.mean(val_mae_list)
        gap = abs(train_mae - val_mae)
        objective = val_mae + alpha * gap

        print(f"Params={params} → Train_MAE={train_mae:.3f}, "
              f"Val_MAE={val_mae:.3f}, Gap={gap:.3f}, Obj={objective:.3f}")

        if objective < best_score:
            best_score = objective
            best_params = params
            best_results = {
                "Train_MAE": train_mae,
                "Val_MAE": val_mae,
                "Gap": gap,
                "Objective": objective
            }

    print(f"\n🏆 Best Params: {best_params}")
    print(f"📉 Best Objective = {best_score:.3f}\n")
    return best_params, best_results


# =========================================================
# 🧠 Tune Top 3 Models → Save with Scaler + Imputer
# =========================================================
def tune_top3_models(df, target="Average_Price"):
    results_path = os.path.join(results_dir, "top3_models.csv")
    if not os.path.exists(results_path):
        alt_path = os.path.join(results_dir, "top3_models_leakfree.csv")
        if os.path.exists(alt_path):
            results_path = alt_path
        else:
            raise FileNotFoundError("⚠️ top3_models.csv not found — run model_pipeline.py first!")

    top3 = pd.read_csv(results_path)
    print(f"🔍 Tuning top 3 models:\n{top3[['Model', 'Val_MAE']]}")

    X_full = df.drop(columns=[target, "Date"], errors="ignore")
    y_full = df[target]

    tuned_results = []
    today = date.today().isoformat()

    # 🧹 Clear old tuned models
    for f in os.listdir(models_dir):
        if f.endswith(".joblib"):
            os.remove(os.path.join(models_dir, f))
    print("🧹 Cleared old tuned models.")

    param_grids = {
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
        },
        "gradient_boost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
        "xgboost": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [6, 8],
        },
    }

    for _, row in top3.iterrows():
        name = row["Model"]
        print(f"\n🚀 Tuning model: {name}")

        if name == "random_forest":
            model_class = RandomForestRegressor
        elif name == "gradient_boost":
            model_class = GradientBoostingRegressor
        elif name == "xgboost" and XGBRegressor:
            model_class = XGBRegressor
        else:
            print(f"⚠️ Skipping unsupported model: {name}")
            continue

        best_params, metrics = tune_model(model_class, param_grids[name], X_full, y_full)

        # ✅ Final training with custom lag & rolling features
        df_full = add_lag_and_rolling(df.copy(),
                                      lag_config=custom_lag_config,
                                      roll_config=custom_roll_config)
        X = df_full.drop(columns=[target, "Date"], errors="ignore")
        y = df_full[target]

        final_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model_class(**best_params))
        ])
        final_pipeline.fit(X, y)

        # Extract & Save
        imputer = final_pipeline.named_steps["imputer"]
        scaler = final_pipeline.named_steps["scaler"]
        model_filename = f"{name}_tuned_latest.joblib"
        model_path = os.path.join(models_dir, model_filename)

        joblib.dump(final_pipeline, model_path)
        joblib.dump(imputer, model_path.replace(".joblib", "_imputer.joblib"))
        joblib.dump(scaler, model_path.replace(".joblib", "_scaler.joblib"))
        joblib.dump(X.columns.tolist(), model_path.replace(".joblib", "_features.joblib"))

        print(f"💾 Saved tuned model → {model_path}")
        print(f"📦 Saved imputer/scaler/features")

        tuned_results.append({
            "Date": today,
            "Model": name,
            "Best_Params": best_params,
            "Train_MAE": round(metrics["Train_MAE"], 3),
            "Val_MAE": round(metrics["Val_MAE"], 3),
            "Gap": round(metrics["Gap"], 3),
            "Objective": round(metrics["Objective"], 3),
            "Model_Path": model_path
        })

    tuned_df = pd.DataFrame(tuned_results)
    tuned_path = os.path.join(results_dir, "tuned_models.csv")
    tuned_df.to_csv(tuned_path, index=False)
    print(f"\n✅ Hyperparameter tuning complete → {tuned_path}")
    print(tuned_df)
    return tuned_df


# =========================================================
# 🚀 Main Execution
# =========================================================
if __name__ == "__main__":
    print("📥 Loading dataset for tuning...")
    df = load_data()

    print("🎯 Starting generalization-aware tuning with custom lag/roll configs...")
    tuned_df = tune_top3_models(df)

    print("\n🏁 Done — tuned pipelines + scaler + imputer saved successfully!")
