import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc

# ======================================================
# 📦 Import Component Modules
# ======================================================
from dash_app.components.data_collection_preprocessing import (
    data_collection_preprocessing_layout,
    register_data_collection_preprocessing_callbacks,
)
from dash_app.components.training_section import (
    training_layout,
    register_training_callbacks,
)
from dash_app.components.prediction_section import (
    prediction_layout,
    register_prediction_callbacks,
)

# Import About Us layout
from dash_app.components.about_us import about_us_layout

# ======================================================
# 🌐 Initialize Dash App
# ======================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    suppress_callback_exceptions=True
)
app.title = "🍅 Kalimati Tomato Forecasting Dashboard"
server = app.server  # For deployment (Render, Heroku, etc.)

# ======================================================
# 🧭 Navbar Component
# ======================================================
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("🏠 Dashboard", href="/")),
        dbc.NavItem(dbc.NavLink("🔮 Prediction", href="/prediction")),
        dbc.NavItem(dbc.NavLink("📄 About Us", href="/about")),  # About Us page link
    ],
    brand="🍅 Kalimati Tomato Forecasting",
    brand_href="/",
    color="danger",
    dark=True,
    sticky="top",
    className="mb-4 shadow-sm px-3",
)

# ======================================================
# 📊 Dashboard Page Layout (2 Columns)
# ======================================================
dashboard_layout = dbc.Container(
    [
        dbc.Row(
            [
                # 🧩 1️⃣ Data Collection & Preprocessing
                dbc.Col(
                    html.Div(
                        data_collection_preprocessing_layout,
                        className="p-3 shadow-sm rounded bg-light",
                    ),
                    width=6,
                ),

                # 🧩 2️⃣ Model Training Section
                dbc.Col(
                    html.Div(
                        training_layout,
                        className="p-3 shadow-sm rounded bg-white",
                    ),
                    width=6,
                ),
            ],
            justify="around",
            align="start",
            className="g-3",
        ),

        html.Hr(),
        html.Footer(
            [
                html.P(
                    "Developed by Kalimati Next Gen AI Team",
                    className="text-center text-muted mb-0",
                )
            ],
            className="mt-3",
        ),
    ],
    fluid=True,
)

# ======================================================
# 🧭 Multi-Page Layout Wrapper
# ======================================================
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        navbar,
        html.Div(id="page-content", className="p-3"),
    ]
)

# ======================================================
# 🔁 Page Routing
# ======================================================
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    """Handles routing between pages."""
    if pathname == "/prediction":
        return prediction_layout
    elif pathname == "/about":
        return about_us_layout()  # About Us page layout
    else:
        return dashboard_layout

# ======================================================
# 🔁 Register All Callbacks
# ======================================================
register_data_collection_preprocessing_callbacks(app)
register_training_callbacks(app)
register_prediction_callbacks(app)

# ======================================================
# 🚀 Run the App
# ======================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
