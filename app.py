import streamlit as st
import plotly.express as px
import pandas as pd
import sys

# Startup confirmation for AWS logs
print("🚀 Streamlit CI/CD deployment started successfully...", file=sys.stderr)

# Page configuration
st.set_page_config(
    page_title="Tomato Price Prediction Dashboard",
    page_icon="🍅",
    layout="wide",
)

# Title
st.title("🍅 Tomato Price Prediction - Auto Deployment Test by Nabin ")

st.markdown("""
This version was automatically **deployed via GitHub Actions** to AWS EC2!  
Use the sidebar controls to explore different **years and continents**.
""")

# Load dataset
df = px.data.gapminder()

# Sidebar
st.sidebar.header("⚙️ Control Panel")

year = st.sidebar.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    2007
)

continent = st.sidebar.selectbox(
    "Select Continent",
    ["All"] + sorted(df["continent"].unique().tolist())
)

# Filter data
if continent != "All":
    filtered_df = df[(df["year"] == year) & (df["continent"] == continent)]
else:
    filtered_df = df[df["year"] == year]

# Handle empty dataset
if filtered_df.empty:
    st.warning("⚠️ No data available for this selection.")
else:
    fig = px.scatter(
        filtered_df,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
        title=f"🌍 Life Expectancy vs GDP per Capita ({year})",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 View Data Sample"):
        st.dataframe(filtered_df.head())

st.markdown("---")
st.caption("✅ Deployed automatically using **GitHub Actions + AWS EC2**")
