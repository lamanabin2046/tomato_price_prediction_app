import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import joblib
import os
import requests
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

# ======================================================
# 🧭 Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "outputs/models/tuned_models")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/tomato_clean_data.csv")

# ======================================================
# 🧩 Helpers
# ======================================================
def add_lag_and_rolling(df, target_col="Average_Price", lags=[1, 3, 7], windows=[7, 14]):
    """Add lag and rolling features for the target column."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    for window in windows:
        df[f"{target_col}_roll{window}"] = df[target_col].rolling(window).mean()
    return df.dropna().reset_index(drop=True)


def safe_predict(model, X):
    """Predict safely — works for both pipelines and plain models."""
    try:
        return model.predict(X)
    except Exception:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return model.predict(X_scaled)


def fetch_all_districts_forecast(days=15):
    """Fetch live weather forecasts for all four districts."""
    DISTRICTS = {
        "Dhading": {"lat": 27.8667, "lon": 84.9167},
        "Kathmandu": {"lat": 27.7172, "lon": 85.3240},
        "Kavre": {"lat": 27.6240, "lon": 85.5475},
        "Sarlahi": {"lat": 26.9833, "lon": 85.5500},
    }

    def fetch_district_forecast(lat, lon, name):
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
            f"&timezone=Asia/Kathmandu&forecast_days={days}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("daily", {})
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["time"])
        df["Temperature"] = df[["temperature_2m_max", "temperature_2m_min"]].mean(axis=1)
        df.rename(columns={"precipitation_sum": f"{name}_Rainfall_MM"}, inplace=True)
        df[f"{name}_Temperature"] = df["Temperature"]
        return df[["Date", f"{name}_Temperature", f"{name}_Rainfall_MM"]]

    frames = []
    for name, coords in DISTRICTS.items():
        try:
            print(f"🌦️ Fetching {name} forecast...")
            frames.append(fetch_district_forecast(coords["lat"], coords["lon"], name))
        except Exception as e:
            print(f"⚠️ {name} fetch failed: {e}")

    if not frames:
        return pd.DataFrame()

    df_final = frames[0]
    for df in frames[1:]:
        df_final = pd.merge(df_final, df, on="Date", how="outer")

    return df_final.sort_values("Date")


def generate_shap_plot(model, X_sample):
    """Generate SHAP feature importance summary plot for the trained model."""
    try:
        final_model = model.named_steps["model"]
    except Exception:
        final_model = model

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.Explainer(final_model.predict, X_sample)
        shap_values = explainer(X_sample)

    plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
    plt.title("Feature Impact on Predicted Tomato Prices")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64,{}".format(encoded)

# ======================================================
# 🔮 Layout
# ======================================================
prediction_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("🔮 Tomato Price Prediction & Forecasting",
                        className="fw-bold text-dark mb-0"), width="auto"),
        dbc.Col([
            html.Div([
                html.Span("⚙️ Choose a Trained Model:", className="fw-bold me-2"),
                dcc.Dropdown(id="prediction-model-dropdown",
                             placeholder="Select a tuned model",
                             style={"width": "260px", "display": "inline-block"},
                             clearable=False),
                dbc.Button("🚀 Run Prediction", id="btn-run-prediction",
                           color="primary", className="ms-2",
                           style={"verticalAlign": "middle"})
            ], style={"display": "flex", "alignItems": "center"})
        ], width="auto", className="ms-auto")
    ], justify="between", align="center", className="mt-2 mb-2 px-3"),

    html.Div(id="prediction-status", className="text-success small mb-2 px-3"),
    html.Hr(className="mt-2 mb-3"),

    dbc.Row([
        # ---------------- LEFT SIDE ----------------
        dbc.Col([
            html.H5("📊 Actual vs Predicted Tomato Prices", className="fw-bold text-primary mb-3"),
            dash_table.DataTable(
                id="prediction-table",
                columns=[
                    {"name": "SN", "id": "SN"},
                    {"name": "Date", "id": "Date"},
                    {"name": "Actual Price", "id": "Actual Price"},
                    {"name": "Predicted Price", "id": "Predicted Price"},
                ],
                style_table={"height": "260px", "overflowY": "auto",
                             "borderRadius": "10px", "border": "1px solid #ddd"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold",
                              "textAlign": "center", "borderBottom": "2px solid #ccc"},
                style_cell={"textAlign": "center", "padding": "5px", "fontSize": "12px"},
                page_size=10
            ),

            html.Hr(className="my-3"),

            html.H6("🌦️ Weather Forecast Summary", className="fw-bold text-success mb-2"),
            dash_table.DataTable(
                id="weather-forecast-table",
                columns=[
                    {"name": "Date", "id": "Date"},
                    {"name": "Dhading_Temp", "id": "Dhading_Temperature"},
                    {"name": "Dhading_Rain", "id": "Dhading_Rainfall_MM"},
                    {"name": "Kath_Temp", "id": "Kathmandu_Temperature"},
                    {"name": "Kath_Rain", "id": "Kathmandu_Rainfall_MM"},
                    {"name": "Kavre_Temp", "id": "Kavre_Temperature"},
                    {"name": "Kavre_Rain", "id": "Kavre_Rainfall_MM"},
                    {"name": "Sarlahi_Temp", "id": "Sarlahi_Temperature"},
                    {"name": "Sarlahi_Rain", "id": "Sarlahi_Rainfall_MM"},
                ],
                style_table={"height": "220px", "overflowY": "auto",
                             "borderRadius": "10px", "border": "1px solid #ddd"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold",
                              "textAlign": "center", "borderBottom": "2px solid #ccc"},
                style_cell={"textAlign": "center", "padding": "5px", "fontSize": "11px"},
                page_size=7
            ),

            html.Hr(className="my-3"),
            html.H5("🧠 Explainable AI: SHAP Feature Impact", className="fw-bold text-info mb-2"),
            html.P("Understand which features (Temperature, Rainfall, Lag, etc.) drive predictions.",
                   className="text-muted small mb-2"),
            dbc.Button("🔍 Generate SHAP Explanation", id="btn-shap", color="info", className="w-100 mb-3"),
            html.Div(id="shap-status", className="text-secondary small mb-2"),
            html.Img(id="shap-plot", style={"width": "100%", "borderRadius": "10px",
                                            "boxShadow": "0 0 6px rgba(0,0,0,0.15)"})
        ], width=4, className="p-3 bg-white rounded shadow-sm"),

        # ---------------- RIGHT SIDE ----------------
        dbc.Col([
            html.H5("📈 Actual vs Predicted Price Trend", className="fw-bold text-secondary mb-3"),
            dcc.Loading(
                id="loading-prediction",
                type="circle",
                color="#007bff",
                fullscreen=False,
                children=dcc.Graph(id="actual-vs-predicted-chart", style={"height": "300px"}),
            ),

            html.Hr(className="my-3"),

            html.H5("⏳ Future Forecast (Weather-Aware)", className="fw-bold mb-3 text-success"),
            html.P("Generate forecasts using live weather data for all 4 districts.",
                   className="text-muted small mb-2"),
            dcc.Dropdown(id="forecast-days-dropdown",
                         options=[
                             {"label": "Next 7 Days", "value": 7},
                             {"label": "Next 15 Days", "value": 15},
                             {"label": "Next 30 Days", "value": 30}],
                         value=7, clearable=False, className="mb-3"),
            dbc.Button("📅 Generate Forecast", id="btn-forecast", color="success", className="w-100 mb-3"),

            html.Div(id="forecast-status", className="text-info small mb-3"),
            dcc.Loading(
                id="loading-forecast",
                type="circle",
                color="#28a745",
                fullscreen=False,
                children=dcc.Graph(id="forecast-chart", style={"height": "300px"}),
            ),

            html.Hr(className="my-3"),
            html.H5("📅 Forecasted Tomato Prices", className="fw-bold text-success mb-3"),
            dash_table.DataTable(
                id="forecast-table",
                columns=[{"name": "SN", "id": "SN"}, {"name": "Date", "id": "Date"},
                         {"name": "Forecasted Price", "id": "Forecasted Price"}],
                style_table={"height": "200px", "overflowY": "auto",
                             "borderRadius": "10px", "border": "1px solid #ddd"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold",
                              "textAlign": "center", "borderBottom": "2px solid #ccc"},
                style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                page_size=7
            )
        ], width=8, className="p-3 bg-light rounded shadow-sm")
    ], className="g-3", justify="center", align="start")
], fluid=True, className="mt-1")


# ======================================================
# ⚙️ Register Callbacks
# ======================================================
def register_prediction_callbacks(app):
    # ✅ Your existing callbacks remain unchanged...

    # 🧠 SHAP Callback
    @app.callback(
        Output("shap-status", "children"),
        Output("shap-plot", "src"),
        Input("btn-shap", "n_clicks"),
        State("prediction-model-dropdown", "value"),
        prevent_initial_call=True
    )
    def explain_model(_, model_name):
        if not model_name:
            return "⚠️ Select a model first.", dash.no_update

        model_path = os.path.join(MODEL_DIR, model_name)
        feature_path = model_path.replace(".joblib", "_features.joblib")

        if not os.path.exists(model_path):
            return f"❌ Model not found: {model_path}", dash.no_update

        model = joblib.load(model_path)
        features = joblib.load(feature_path) if os.path.exists(feature_path) else None

        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Average_Price"])
        df = add_lag_and_rolling(df)
        X = df.drop(columns=["Average_Price", "Date"], errors="ignore")

        if features:
            for col in features:
                if col not in X.columns:
                    X[col] = 0
            X = X[features]

        X_sample = X.sample(min(200, len(X)), random_state=42)

        try:
            shap_image = generate_shap_plot(model, X_sample)
            return f"✅ SHAP explanation generated for {model_name}.", shap_image
        except Exception as e:
            return f"⚠️ SHAP generation failed: {e}", dash.no_update
