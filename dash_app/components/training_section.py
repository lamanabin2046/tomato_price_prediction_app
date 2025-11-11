import dash
from dash import html, Output, Input, ctx, dcc
import dash_bootstrap_components as dbc
import subprocess
from datetime import datetime
import os
import pandas as pd
import json
import time

# ======================================================
# 📁 Absolute Paths
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs/results")
MODELS_DIR = os.path.join(BASE_DIR, "outputs/models")


# ======================================================
# 🧩 Layout
# ======================================================
training_layout = html.Div([
    html.H5("⚙️ Model Operations", className="mb-3"),

    dbc.Button("📊 Run Cross-Validation", id="btn-train", color="primary", className="mb-2"),
    dbc.Button("🔍 Run Hyperparameter Tuning", id="btn-tune", color="warning", className="mb-2 me-2"),

    dcc.Loading(
        id="loading",
        type="circle",
        color="#007BFF",
        children=html.Div(id="train-status", className="mt-3 text-info"),
        style={"transform": "scale(1.5)", "marginTop": "20px"}
    )
])


# ======================================================
# ⚙️ Register Callbacks
# ======================================================
def register_training_callbacks(app):
    @app.callback(
        Output("train-status", "children"),
        Input("btn-tune", "n_clicks"),
        Input("btn-train", "n_clicks"),
        prevent_initial_call=True
    )
    def run_operations(btn_tune, btn_train):
        triggered = ctx.triggered_id
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ------------------------------------------------------
        # 🔍 Run Hyperparameter Tuning
        # ------------------------------------------------------
        if triggered == "btn-tune":
            print("🚀 Running hyperparameter tuning subprocess...")
            subprocess.run(["python", "src/modeling/hyperparameter_tuning.py"], cwd=BASE_DIR)

            tuned_candidates = [
                os.path.join(RESULTS_DIR, "tuned_models_leakfree.csv"),
                os.path.join(RESULTS_DIR, "tuned_models.csv")
            ]
            tuned_path = next((p for p in tuned_candidates if os.path.exists(p)), None)

            for _ in range(5):
                if tuned_path and os.path.exists(tuned_path):
                    break
                time.sleep(1)

            if not tuned_path:
                print("❌ Tuned models not found:", tuned_candidates)
                return html.Div("⚠️ No tuned models found. Please check logs.", className="text-danger")

            print(f"✅ Found tuned model file at: {tuned_path}")
            tuned_df = pd.read_csv(tuned_path)

            if tuned_df.empty:
                return html.Div("⚠️ Tuned models file is empty.", className="text-danger")

            if "Best_Params" in tuned_df.columns:
                tuned_df["Best_Params"] = tuned_df["Best_Params"].apply(
                    lambda x: json.dumps(eval(x), indent=2)
                    if isinstance(x, str) and x.strip().startswith("{")
                    else x
                )

            available_cols = [
                c for c in ["Model", "Train_MAE", "Val_MAE", "Gap", "Objective", "Best_Params", "Model_Path"]
                if c in tuned_df.columns
            ]

            tuned_table = html.Div([
                dbc.Table.from_dataframe(
                    tuned_df[available_cols],
                    striped=True, bordered=True, hover=True,
                    size="sm", className="mt-3 table-warning"
                )
            ], style={
                "maxHeight": "320px",
                "overflowY": "auto",
                "fontSize": "10px",  # 👈 reduced font size (was 12px)
                "fontFamily": "monospace",
                "padding": "4px"
            })

            return html.Div([
                html.H6("🏆 Hyperparameter Tuning Completed", className="fw-bold text-success"),
                html.P(f"✅ Completed at {timestamp}", className="text-muted small"),
                html.P(f"📄 File: {os.path.basename(tuned_path)}", className="text-secondary small"),
                html.Hr(),
                html.H6("📊 Tuned Models Summary:", className="fw-bold text-primary"),
                tuned_table
            ])

        # ------------------------------------------------------
        # 📊 Cross Validation (Model Training)
        # ------------------------------------------------------
        elif triggered == "btn-train":
            print("🚀 Running model pipeline subprocess...")
            subprocess.run(["python", "src/modeling/model_pipeline.py"], cwd=BASE_DIR)

            results_candidates = [
                os.path.join(RESULTS_DIR, "cv_results_leakfree.csv"),
                os.path.join(RESULTS_DIR, "cv_results.csv")
            ]
            top3_candidates = [
                os.path.join(RESULTS_DIR, "top3_models_leakfree.csv"),
                os.path.join(RESULTS_DIR, "top3_models.csv")
            ]

            results_path = next((p for p in results_candidates if os.path.exists(p)), None)
            top3_path = next((p for p in top3_candidates if os.path.exists(p)), None)

            if not results_path:
                return html.Div("⚠️ No results found. Please check logs.", className="text-danger")

            print(f"✅ Found results file: {results_path}")
            df_all = pd.read_csv(results_path)

            latest_date = df_all["Date"].max() if "Date" in df_all.columns else None
            df_latest = df_all[df_all["Date"] == latest_date].copy() if latest_date else df_all.copy()

            df_top3 = pd.read_csv(top3_path) if top3_path and os.path.exists(top3_path) else None

            if "Val_MAE" not in df_latest.columns:
                return html.Div("⚠️ Missing Val_MAE column in results.", className="text-danger")

            best_row = df_latest.loc[df_latest["Val_MAE"].idxmin()]
            best_model = best_row.get("Model", "Unknown")
            val_mae = best_row.get("Val_MAE", None)
            val_r2 = best_row.get("Val_R2", None)

            # 🥇 Top 3 Table
            if df_top3 is not None and not df_top3.empty:
                cols_top3 = [c for c in ["Model", "Val_MAE", "Val_R2"] if c in df_top3.columns]
                top3_table = dbc.Table.from_dataframe(
                    df_top3[cols_top3],
                    striped=True, bordered=True, hover=True,
                    size="sm", className="mt-3 table-success"
                )
            else:
                top3_table = html.P("⚠️ No top 3 file found.", className="text-danger")

            # 📋 All Model Results (smaller font)
            available_cols = [
                c for c in ["Model", "Train_MAE", "Val_MAE", "Train_R2", "Val_R2", "Val_MAE_std"]
                if c in df_latest.columns
            ]
            all_table = html.Div([
                dbc.Table.from_dataframe(
                    df_latest[available_cols],
                    striped=True, bordered=True, hover=True,
                    size="sm", className="mt-3"
                )
            ], style={
                "fontSize": "10px",  # 👈 reduced font size
                "fontFamily": "monospace",
                "maxHeight": "300px",
                "overflowY": "auto",
                "padding": "4px"
            })

            summary = html.Div([
                html.H6(f"🏆 Best Model: {best_model}", className="fw-bold text-success"),
                html.P(
                    f"📉 Val MAE: {val_mae:.3f} | 📈 Val R²: {val_r2:.3f}" if val_r2 is not None
                    else f"📉 Val MAE: {val_mae:.3f}",
                    className="text-secondary"
                ),
                html.P(f"📄 File: {os.path.basename(results_path)}", className="text-secondary small"),
                html.Hr(),
                html.H6("🥇 Top 3 Models (Lowest Val MAE):", className="fw-bold text-primary"),
                html.Div(top3_table, style={"fontSize": "10px"}),  # 👈 smaller font for Top 3 table
                html.Br(),
                html.H6("📋 All Model Cross-Validation Results:", className="fw-bold text-primary"),
                all_table,
                html.P(f"✅ Validation completed at {timestamp}", className="text-muted small mt-3")
            ])

            return summary
