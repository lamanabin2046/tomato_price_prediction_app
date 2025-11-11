import dash
from dash import html, dcc, Output, Input, State, dash_table, ctx
import dash_bootstrap_components as dbc
import subprocess
from datetime import datetime
import os
import base64
import pandas as pd

UPLOAD_FOLDER = "data/raw"

# ======================================================
# 📘 Project Description Card
# ======================================================
project_info_card = dbc.Card([
    dbc.Button(
        "📘 About Project, Data & Kalimati Market",
        id="toggle-info",
        color="danger",
        className="mb-2 w-100 fw-bold",
        style={"fontSize": "15px"}
    ),
    dbc.Collapse(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("🍅 Kalimati Tomato Market Forecasting System",
                            className="fw-bold text-danger mb-1"),
                    html.P(
                        "This dashboard forecasts tomato prices in Nepal’s Kalimati Market using AI models "
                        "trained on weather, economic, and agricultural indicators.",
                        className="text-muted fst-italic mb-3",
                        style={"fontSize": "13px"}
                    ),
                    html.H6("📊 Data Collection Methodology", className="fw-bold text-primary mt-2"),
                    html.Ul([
                        html.Li("Web scraping from Kalimati Market’s daily price and supply data."),
                        html.Li("Weather data fetched from Open-Meteo API."),
                        html.Li("Supplementary inputs: fuel, inflation, and exchange rate datasets."),
                        html.Li("Preprocessing merges, cleans, and formats data for modeling.")
                    ], className="text-muted", style={"fontSize": "13px"}),

                    html.H6("🏪 About Kalimati Market", className="fw-bold text-success mt-3"),
                    html.P(
                        """
                        The Kalimati Fruits and Vegetable Market is Nepal’s largest wholesale agricultural hub.
                        It serves as the primary reference for vegetable prices across Nepal. Monitoring its data 
                        allows data-driven forecasting and smarter supply-chain management.
                        """,
                        className="text-muted",
                        style={"fontSize": "13px", "textAlign": "justify"}
                    )
                ], width=8),

                dbc.Col([
                    html.Img(
                        src="https://upload.wikimedia.org/wikipedia/commons/5/52/Tomatoes_je.jpg",
                        style={
                            "width": "100%", "maxHeight": "160px",
                            "objectFit": "cover", "borderRadius": "10px",
                            "boxShadow": "0 2px 6px rgba(0,0,0,0.2)", "marginBottom": "8px"
                        }
                    ),
                    html.Img(
                        src="https://nepalitimes.com/media/albums/Kalimati_Market.original.jpg",
                        style={
                            "width": "100%", "maxHeight": "160px",
                            "objectFit": "cover", "borderRadius": "10px",
                            "boxShadow": "0 2px 6px rgba(0,0,0,0.2)"
                        }
                    )
                ], width=4)
            ])
        ]),
        id="collapse-info",
        is_open=False
    )
], className="shadow-sm bg-white rounded mb-4")

# ======================================================
# 📤 Upload & Data Preprocessing Layout
# ======================================================
data_collection_preprocessing_layout = html.Div([
    project_info_card,

    html.H5("📤 Upload & Data Preprocessing", className="fw-bold text-primary mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-fuel",
                children=html.Div([
                    html.Span("⛽ Fuel Price CSV", className="fw-bold text-secondary"),
                    html.Br(),
                    html.Span("Upload .csv file", style={"fontSize": "11px", "color": "#888"})
                ]),
                className="upload-box mb-2", multiple=False
            )
        ], width=4),

        dbc.Col([
            dcc.Upload(
                id="upload-inflation",
                children=html.Div([
                    html.Span("💰 Inflation XLSX", className="fw-bold text-secondary"),
                    html.Br(),
                    html.Span("Upload .xlsx file", style={"fontSize": "11px", "color": "#888"})
                ]),
                className="upload-box mb-2", multiple=False
            )
        ], width=4),

        dbc.Col([
            dcc.Upload(
                id="upload-exchange",
                children=html.Div([
                    html.Span("💱 Exchange Rate CSV", className="fw-bold text-secondary"),
                    html.Br(),
                    html.Span("Upload .csv file", style={"fontSize": "11px", "color": "#888"})
                ]),
                className="upload-box mb-2", multiple=False
            )
        ], width=4)
    ], className="mb-2"),

    dbc.ButtonGroup([
        dbc.Button("📥 Collect Data (Scraping)", id="btn-scrape-data", color="info", className="me-2"),
        dbc.Button("🧮 Preprocess Data", id="btn-preprocess-data", color="success")
    ], className="mb-3"),

    dcc.Loading(
        id="loading-scrape",
        type="circle",
        color="#007BFF",
        style={"transform": "scale(1.1)", "marginTop": "15px"},
        children=html.Div([
            html.Div(id="data-collection-status", className="mt-3 text-info"),
            html.Div(id="upload-status", className="mt-2 text-success"),
            html.H6("🔍 Sample Data Preview:", className="fw-bold text-secondary mt-3"),
            dash_table.DataTable(
                id="data-display-table",
                data=[],
                columns=[],
                style_table={
                    'height': '280px',
                    'overflowY': 'auto',
                    'border': '1px solid #ddd',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'marginBottom': '10px'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '13px',
                    'whiteSpace': 'normal',
                    'borderBottom': '1px solid #eee'
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#f8f9fa',
                    'borderBottom': '2px solid #ccc'
                }
            )
        ])
    )
])

# ======================================================
# ⚙️ Helper to Run Scripts and Capture Logs
# ======================================================
def run_script_and_capture(script_path):
    logs = []
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["chcp"] = "65001"  # Windows encoding fix

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        stdout, stderr = process.communicate()
        if stdout:
            logs.extend(stdout.splitlines())
        if stderr:
            logs.append("\n⚠️ [ERROR LOGS BELOW]")
            logs.extend(stderr.splitlines())
        if process.returncode != 0:
            logs.append(f"\n❌ Script failed with exit code {process.returncode}")
        else:
            logs.append(f"✅ {os.path.basename(script_path)} completed successfully.")
    except Exception as e:
        logs.append(f"❌ Exception while running {script_path}: {e}")
    return logs

# ======================================================
# ⚙️ Callback Registration
# ======================================================
def register_data_collection_preprocessing_callbacks(app):

    # 🟢 Toggle Description Panel
    @app.callback(
        Output("collapse-info", "is_open"),
        Input("toggle-info", "n_clicks"),
        State("collapse-info", "is_open"),
        prevent_initial_call=True
    )
    def toggle_info(n, is_open):
        return not is_open if n else is_open

    # 🧩 Upload, Scrape, and Preprocess Logic
    @app.callback(
        [Output("data-collection-status", "children"),
         Output("upload-status", "children"),
         Output("data-display-table", "data"),
         Output("data-display-table", "columns")],
        [Input("upload-fuel", "contents"),
         Input("upload-inflation", "contents"),
         Input("upload-exchange", "contents"),
         Input("btn-scrape-data", "n_clicks"),
         Input("btn-preprocess-data", "n_clicks")],
        [State("upload-fuel", "filename"),
         State("upload-inflation", "filename"),
         State("upload-exchange", "filename")]
    )
    def upload_and_process(fuel_content, inflation_content, exchange_content,
                           btn_scrape_data, btn_preprocess_data,
                           fuel_name, inf_name, exch_name):

        triggered = ctx.triggered_id
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uploaded, df_display, df_columns = [], [], []

        # 🟦 Default (load sample data)
        if triggered is None:
            try:
                df = pd.read_csv("data/processed/tomato_clean_data.csv", encoding="utf-8").tail(5)
                df_display = df.to_dict("records")
                df_columns = [{"name": i, "id": i} for i in df.columns]
                return html.P("📊 Showing sample data (default view).", className="text-muted"), dash.no_update, df_display, df_columns
            except Exception:
                return html.Div("⚠️ No data available."), dash.no_update, [], []

        # 🟨 File Uploads
        if triggered in ["upload-fuel", "upload-inflation", "upload-exchange"]:
            for content, name in zip(
                [fuel_content, inflation_content, exchange_content],
                [fuel_name, inf_name, exch_name]
            ):
                if content:
                    content_type, content_string = content.split(",")
                    decoded = base64.b64decode(content_string)
                    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                    save_path = os.path.join(UPLOAD_FOLDER, name)
                    with open(save_path, "wb") as f:
                        f.write(decoded)
                    uploaded.append(name)
            upload_status = f"✅ Uploaded files: {', '.join(uploaded)}"
            return dash.no_update, upload_status, dash.no_update, dash.no_update

        # 🟩 Data Scraping
        elif triggered == "btn-scrape-data":
            try:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
                scrapers = [
                    os.path.join(PROJECT_ROOT, "src", "scrapers", "scraper_arrival.py"),
                    os.path.join(PROJECT_ROOT, "src", "scrapers", "scraper_price.py"),
                    os.path.join(PROJECT_ROOT, "src", "scrapers", "weather.py")
                ]

                logs = ["⏳ Collecting data...\n"]
                for script in scrapers:
                    if os.path.exists(script):
                        logs.append(f"🚀 Running {os.path.basename(script)} ...")
                        logs.extend(run_script_and_capture(script))
                    else:
                        logs.append(f"⚠️ Missing: {os.path.basename(script)}")

                return (
                    html.Pre("\n".join(logs),
                             style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "border": "1px solid #ddd",
                                    "padding": "10px", "borderRadius": "8px", "fontSize": "13px", "maxHeight": "250px",
                                    "overflowY": "auto"}),
                    dash.no_update, [], []
                )
            except Exception as e:
                return html.Div(f"❌ Scraping error: {e}", className="text-danger fw-bold"), dash.no_update, [], []

        # 🟧 Data Preprocessing
        elif triggered == "btn-preprocess-data":
            try:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
                build_script = os.path.join(PROJECT_ROOT, "src", "preprocessing", "build_dataset.py")
                logs = run_script_and_capture(build_script)

                df = pd.read_csv("data/processed/tomato_clean_data.csv", encoding="utf-8").tail(5)
                df_display = df.to_dict("records")
                df_columns = [{"name": i, "id": i} for i in df.columns]

                return (
                    html.Div([
                        html.H6("🧮 Preprocessing Logs:", className="fw-bold mt-2"),
                        html.Pre("\n".join(logs),
                                 style={"whiteSpace": "pre-wrap", "backgroundColor": "#f8f9fa", "border": "1px solid #ddd",
                                        "padding": "10px", "borderRadius": "8px", "fontSize": "13px", "maxHeight": "250px",
                                        "overflowY": "auto"}),
                        html.P(f"✅ Preprocessing completed at {timestamp}", className="text-success fw-bold mt-2")
                    ]),
                    dash.no_update, df_display, df_columns
                )

            except Exception as e:
                return html.Div(f"❌ Preprocessing error: {e}", className="text-danger fw-bold"), dash.no_update, [], []
