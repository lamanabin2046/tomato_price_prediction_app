import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import joblib
import os
import requests
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# ======================================================
# 🧭 Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "outputs/models/tuned_models")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/tomato_clean_data.csv")
WEATHER_PATH = os.path.join(BASE_DIR, "data/processed/weather_forecast.csv")

# ======================================================
# ⚙️ Custom Lag & Rolling Configurations
# ======================================================
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

# ======================================================
# 🧩 Helpers
# ======================================================
def add_lag_and_rolling(df, lag_config=None, roll_config=None):
    df = df.copy()
    lag_config = lag_config or {}
    roll_config = roll_config or {}
    for col, lags in lag_config.items():
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for col, windows in roll_config.items():
        if col in df.columns:
            for window in windows:
                df[f"{col}_roll{window}"] = df[col].rolling(window).mean()
    return df


def safe_predict(model, X, scaler_path=None):
    try:
        return model.predict(X)
    except Exception:
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return model.predict(X_scaled)

# ======================================================
# 🌦️ Weather Fetching & Caching
# ======================================================
def fetch_or_load_weather(days=15):
    os.makedirs(os.path.dirname(WEATHER_PATH), exist_ok=True)
    if os.path.exists(WEATHER_PATH):
        df = pd.read_csv(WEATHER_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        last_date = df["Date"].max().date()
        today = datetime.now().date()
        if (last_date - today).days >= days - 1:
            return df

    DISTRICTS = {
        "Dhading": {"lat": 27.8667, "lon": 84.9167},
        "Kathmandu": {"lat": 27.7172, "lon": 85.3240},
        "Kavre": {"lat": 27.6240, "lon": 85.5475},
        "Sarlahi": {"lat": 26.9833, "lon": 85.5500},
    }

    def fetch_district(lat, lon, name, retries=3):
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=Asia/Kathmandu&forecast_days={days}"
        )
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json().get("daily", {})
                if "time" not in data:
                    continue
                df = pd.DataFrame(data)
                df["Date"] = pd.to_datetime(df["time"])
                df[f"{name}_Temperature"] = df[["temperature_2m_max", "temperature_2m_min"]].mean(axis=1)
                df[f"{name}_Rainfall_MM"] = df["precipitation_sum"]
                return df[["Date", f"{name}_Temperature", f"{name}_Rainfall_MM"]]
            except Exception:
                time.sleep(2)
        print(f"⚠️ Failed to fetch data for {name}. Skipping...")
        return pd.DataFrame(columns=["Date", f"{name}_Temperature", f"{name}_Rainfall_MM"])

    frames = [fetch_district(v["lat"], v["lon"], k) for k, v in DISTRICTS.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("❌ No weather data available from any district API!")

    df_final = frames[0]
    for df in frames[1:]:
        if "Date" in df.columns:
            df_final = pd.merge(df_final, df, on="Date", how="outer")

    if {"Dhading_Temperature", "Kavre_Temperature"}.issubset(df_final.columns):
        df_final["Hilly_Temperature"] = df_final[["Dhading_Temperature", "Kavre_Temperature"]].mean(axis=1)
    if {"Dhading_Rainfall_MM", "Kavre_Rainfall_MM"}.issubset(df_final.columns):
        df_final["Hilly_Rainfall_MM"] = df_final[["Dhading_Rainfall_MM", "Kavre_Rainfall_MM"]].mean(axis=1)

    cols = [
        "Date", "Kathmandu_Temperature", "Kathmandu_Rainfall_MM",
        "Sarlahi_Temperature", "Sarlahi_Rainfall_MM",
        "Hilly_Temperature", "Hilly_Rainfall_MM"
    ]
    cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[cols].dropna(how="all")
    df_final.to_csv(WEATHER_PATH, index=False)
    return df_final

# ======================================================
# 🧠 SHAP Explainability
# ======================================================
def generate_shap_plot(model, X_sample):
    try:
        final_model = model.named_steps.get("model", model)
    except Exception:
        final_model = model
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)
        shap_df = pd.DataFrame(abs(shap_values).mean(axis=0), index=X_sample.columns, columns=["importance"])
    except Exception:
        explainer = shap.Explainer(final_model.predict, X_sample)
        shap_values = explainer(X_sample)
        shap_df = pd.DataFrame(abs(shap_values.values).mean(axis=0), index=X_sample.columns, columns=["importance"])
    shap_df = shap_df.sort_values(by="importance", ascending=False).head(5)
    plt.figure(figsize=(7, 3))
    plt.barh(shap_df.index[::-1], shap_df["importance"].iloc[::-1], color="#007bff")
    plt.title("Top 5 Features Influencing Tomato Prices")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ======================================================
# 🔮 Layout
# ======================================================
prediction_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("🔮 Tomato Price Prediction & Forecasting",
                        className="fw-bold text-dark mb-0"), width="auto"),
        dbc.Col([
            html.Div([
                html.Span("⚙️ Select Model:", className="fw-bold me-2"),
                dcc.Dropdown(id="prediction-model-dropdown",
                             placeholder="Select *_latest.joblib model",
                             style={"width": "300px", "display": "inline-block"},
                             clearable=False),
                html.Span("📅 Forecast Days:", className="fw-bold ms-3 me-2"),
                dcc.Dropdown(id="forecast-days-dropdown",
                             options=[{"label": f"Next {i} Days", "value": i} for i in [7, 15, 30]],
                             value=7, clearable=False, style={"width": "150px"}),
                dbc.Button("🚀 Run Prediction", id="btn-run-prediction",
                           color="success", className="ms-3")
            ], style={"display": "flex", "alignItems": "center"})
        ])
    ]),
    html.Div(id="forecast-status", className="text-success small mb-2 px-3"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5("📅 Past 1-Month Price Summary", className="fw-bold text-primary mb-2"),
            dash_table.DataTable(
                id="past-month-price-table",
                columns=[
                    {"name": "Date", "id": "Date"},
                    {"name": "Actual Price (NPR/kg)", "id": "Actual_Price"},
                    {"name": "Predicted Price (NPR/kg)", "id": "Predicted_Price"}
                ],
                style_table={"height": "240px", "overflowY": "auto"},
                style_cell={"textAlign": "center", "fontSize": "11px"},
            ),
            html.Hr(),
            html.H5("🌦️ Weather Forecast Summary", className="fw-bold text-success mb-2"),
            dash_table.DataTable(
                id="weather-forecast-table",
                style_table={"height": "240px", "overflowY": "auto"},
                style_cell={"textAlign": "center", "fontSize": "11px"}
            ),
            html.Div(id="weather-last-updated", className="text-muted small mt-1"),
            html.Hr(),
            html.H5("🧠 SHAP Feature Impact", className="fw-bold text-info mb-2"),
            html.Div(id="shap-status", className="text-secondary small mb-2"),
            html.Img(id="shap-plot", style={"width": "100%", "borderRadius": "10px",
                                            "boxShadow": "0 0 6px rgba(0,0,0,0.15)"})
        ], width=4, className="p-3 bg-white rounded shadow-sm"),
        dbc.Col([
            html.H5("📈 Forecast Results", className="fw-bold mb-3 text-success"),
            dcc.Graph(id="forecast-chart", style={"height": "320px"}),
            dash_table.DataTable(id="forecast-table",
                                 columns=[{"name": i, "id": i} for i in ["SN", "Date", "Forecasted Price"]],
                                 style_table={"height": "220px", "overflowY": "auto"},
                                 style_cell={"textAlign": "center", "fontSize": "12px"}),
            html.Div(id="price-summary", className="mt-2 small fw-semibold text-dark")
        ], width=8, className="p-3 bg-light rounded shadow-sm")
    ])
], fluid=True, className="mt-1")

# ======================================================
# ⚙️ Callbacks
# ======================================================
def register_prediction_callbacks(app):

    @app.callback(
        Output("prediction-model-dropdown", "options"),
        Output("prediction-model-dropdown", "value"),
        Input("prediction-model-dropdown", "id")
    )
    def load_models(_):
        if not os.path.exists(MODEL_DIR):
            return [], None
        models = [f for f in os.listdir(MODEL_DIR)
                  if f.endswith("_latest.joblib") and "_features" not in f]
        opts = [{"label": m, "value": m} for m in models]
        return opts, (models[0] if models else None)

    @app.callback(
        Output("forecast-status", "children"),
        Output("forecast-chart", "figure"),
        Output("forecast-table", "data"),
        Output("price-summary", "children"),
        Output("past-month-price-table", "data"),
        Output("weather-forecast-table", "data"),
        Output("weather-last-updated", "children"),
        Output("shap-status", "children"),
        Output("shap-plot", "src"),
        Input("btn-run-prediction", "n_clicks"),
        State("forecast-days-dropdown", "value"),
        State("prediction-model-dropdown", "value"),
        prevent_initial_call=True
    )
    def run_full_forecast(_, days, model_name):
        try:
            model_path = os.path.join(MODEL_DIR, model_name)
            scaler_path = model_path.replace(".joblib", "_scaler.joblib")
            feature_path = model_path.replace(".joblib", "_features.joblib")
            model = joblib.load(model_path)
            features = joblib.load(feature_path) if os.path.exists(feature_path) else None

            df = pd.read_csv(DATA_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
            weather_df = fetch_or_load_weather(days)
            last_static = df[["Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation"]].iloc[-1].to_dict()
            last_actual_price = df["Average_Price"].iloc[-1]
            df_full = add_lag_and_rolling(df, custom_lag_config, custom_roll_config)
            results = []

            for i in range(days):
                next_date = df_full["Date"].max() + timedelta(days=1)
                month = next_date.month
                day = next_date.day
                day_of_week = next_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                seasons = {"Season_Winter": 0, "Season_Spring": 0, "Season_Monsoon": 0, "Season_Autumn": 0}
                if month in [12, 1, 2]:
                    seasons["Season_Winter"] = 1
                elif month in [3, 4, 5]:
                    seasons["Season_Spring"] = 1
                elif month in [6, 7, 8, 9]:
                    seasons["Season_Monsoon"] = 1
                else:
                    seasons["Season_Autumn"] = 1
                is_festival = 1 if month in [9, 10, 11] else 0

                row = {
                    "Date": next_date, "day": day, "month": month,
                    "day_of_week": day_of_week, "is_weekend": is_weekend,
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                    "is_festival": is_festival, **last_static, **seasons
                }
                if next_date in weather_df["Date"].values:
                    w = weather_df.loc[weather_df["Date"] == next_date].iloc[0]
                    for c in w.index:
                        if c != "Date":
                            row[c] = w[c]

                df_full = pd.concat([df_full, pd.DataFrame([row])], ignore_index=True)
                df_full = add_lag_and_rolling(df_full, custom_lag_config, custom_roll_config)
                df_latest = df_full.iloc[[-1]].drop(columns=["Average_Price"], errors="ignore")
                if features:
                    for c in features:
                        if c not in df_latest.columns:
                            df_latest[c] = 0
                    df_latest = df_latest[features]

                y_pred = float(safe_predict(model, df_latest, scaler_path)[0])
                if i == 0:
                    y_pred = 0.7 * y_pred + 0.3 * last_actual_price
                df_full.loc[df_full.index[-1], "Average_Price"] = y_pred
                results.append({"SN": i + 1, "Date": next_date.strftime("%Y-%m-%d"),
                                "Forecasted Price": round(y_pred, 2)})

            forecast_df = pd.DataFrame(results)

            # Past Month Actual vs Predicted
            past_month_df = df.tail(30)[["Date", "Average_Price"]].copy()
            past_month_df.rename(columns={"Average_Price": "Actual_Price"}, inplace=True)
            past_month_features = add_lag_and_rolling(df, custom_lag_config, custom_roll_config).dropna().tail(30)
            X_past = past_month_features.drop(columns=["Average_Price", "Date"], errors="ignore")
            if features:
                for c in features:
                    if c not in X_past.columns:
                        X_past[c] = 0
                X_past = X_past[features]
            past_month_df["Predicted_Price"] = safe_predict(model, X_past, scaler_path)
            past_month_df["Date"] = past_month_df["Date"].dt.strftime("%Y-%m-%d")
            past_month_data = past_month_df.to_dict("records")

            # Forecast Chart
            last7_df = df.tail(7)[["Date", "Average_Price"]].copy()
            last7_df["Type"] = "Actual"
            forecast_df["Type"] = "Forecast"
            combined_df = pd.concat([
                last7_df.rename(columns={"Average_Price": "Forecasted Price"}),
                forecast_df[["Date", "Forecasted Price", "Type"]]
            ])
            fig = px.line(combined_df, x="Date", y="Forecasted Price",
                          color="Type", markers=True,
                          title=f"📈 Tomato Price Forecast ({days} Days Ahead)")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (NPR/kg)", template="plotly_white")

            # Price Summary
            actual_last7_mean = df["Average_Price"].tail(7).mean()
            forecast_mean = forecast_df["Forecasted Price"].mean()
            change = ((forecast_mean - actual_last7_mean) / actual_last7_mean) * 100
            price_summary = html.Div([
                html.Hr(style={"margin": "4px 0"}),
                html.Span("🧾 Price Summary (Actual vs Forecast)", className="fw-bold text-primary"),
                html.Div([html.Span("📊 Avg Actual (Last 7d): ", className="text-secondary"),
                          f"{actual_last7_mean:.2f} NPR/kg"]),
                html.Div([html.Span("🔮 Avg Forecast (Next Days): ", className="text-secondary"),
                          f"{forecast_mean:.2f} NPR/kg"]),
                html.Div([html.Span("📉 Expected Change: ", className="text-secondary"),
                          f"{change:+.2f}%"])
            ])

            # SHAP Plot
            X = df_full.drop(columns=["Average_Price", "Date"], errors="ignore").dropna()
            X_sample = X.sample(min(200, len(X)), random_state=42)
            shap_img = generate_shap_plot(model, X_sample)

            return (
                f"✅ Forecast generated using {model_name}.",
                fig,
                forecast_df[["SN", "Date", "Forecasted Price"]].to_dict("records"),
                price_summary,
                past_month_data,
                weather_df.tail(days).to_dict("records"),
                f"📅 Last Updated: {weather_df['Date'].max().strftime('%Y-%m-%d')}",
                f"✅ SHAP generated for {model_name}.",
                shap_img
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"❌ Error: {e}", dash.no_update, [], "", [], [], "", f"⚠️ Error: {e}", dash.no_update)
