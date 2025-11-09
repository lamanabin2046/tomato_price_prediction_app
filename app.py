# ✅ Tomato Price Forecasting App (Fix: Boolean → Int)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

st.set_page_config(page_title="🍅 Tomato Price Forecasting", layout="wide")
st.title("🍅 Tomato Price Forecasting App (ML Models)")

DATA_PATH = "artifacts/data/clean_data_with_lag_roll.csv"
MODEL_DIR = "artifacts/models_lag"
META_PATH = f"{MODEL_DIR}/model_meta.joblib"

# ✅ Load Metadata
meta = joblib.load(META_PATH)
feature_cols = meta["feature_cols"]

# ✅ Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    return df

df = load_data()
st.success(f"✅ Data loaded: {df.shape[0]} rows")

model_files = [m for m in os.listdir(MODEL_DIR)
               if m.endswith(".joblib") and "meta" not in m]

selected_model = st.selectbox("Select Model", model_files)

@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, name))

model = load_model(selected_model)
st.success(f"✅ Model Loaded: {selected_model}")

forecast_days = st.radio("Forecast Days", [7, 15, 30])

# ✅ Predict Button
if st.button("🔮 Predict Prices"):
    try:
        df_copy = df.copy()
        future_dates = []
        preds = []

        last_date = df_copy["date"].iloc[-1]

        for i in range(forecast_days):
            new_date = last_date + timedelta(days=i + 1)
            future_dates.append(new_date)

            new_row = df_copy.iloc[-1:].copy()
            new_row["date"] = new_date

            # ✅ Time-based features
            new_row["day"] = new_row["date"].dt.day
            new_row["month"] = new_row["date"].dt.month
            new_row["day_of_week"] = new_row["date"].dt.dayofweek
            new_row["is_weekend"] = (new_row["day_of_week"] >= 5).astype(int)
            new_row["month_sin"] = np.sin(2 * np.pi * new_row["month"] / 12)
            new_row["month_cos"] = np.cos(2 * np.pi * new_row["month"] / 12)

            # ✅ Drop target if exists
            new_row = new_row.drop(columns=["Average_Price"],
                                   errors="ignore")

            # ✅ Convert boolean → Int (Fix!)
            bool_cols = new_row.select_dtypes(include=["bool"]).columns
            new_row[bool_cols] = new_row[bool_cols].astype(int)

            # ✅ Align features
            new_row = new_row.reindex(columns=feature_cols, fill_value=0)

            pred = model.predict(new_row)[0]
            preds.append(pred)

            # ✅ Append to df history for autoregression
            next_row = df_copy.iloc[-1:].copy()
            next_row["Average_Price"] = pred
            df_copy = pd.concat([df_copy, next_row], ignore_index=True)

        result_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Tomato Price (NPR)": preds
        })

        st.subheader(f"📈 Forecast ({forecast_days} days) using {selected_model}")
        st.line_chart(result_df.set_index("Date"))
        st.dataframe(result_df)

        st.download_button(
            "⬇️ Download CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            f"forecast_{forecast_days}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
