import streamlit as st
import plotly.express as px
import pandas as pd
import sys

# Startup confirmation for AWS EB logs
print("✅ Streamlit app starting successfully...", file=sys.stderr)

# Page configuration
st.set_page_config(
    page_title="Tomato Price Visualization Demo",
    page_icon="🍅",
    layout="wide",
)

# Title
st.title("🍅 Interactive Plotly Chart with Streamlit")

st.markdown("""
This demo shows how to deploy a **Streamlit + Plotly** app on AWS Elastic Beanstalk.
Use the sidebar to select a year and continent to explore global statistics.
""")

# Load sample dataset
df = px.data.gapminder()

# Sidebar controls
st.sidebar.header("🔧 User Controls")

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

# Handle empty results
if filtered_df.empty:
    st.warning("No data available for this selection.")
else:
    # Plot
    fig = px.scatter(
        filtered_df,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
        title=f"Life Expectancy vs GDP per Capita ({year})",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Data preview"):
        st.dataframe(filtered_df.head())

st.markdown("---")
st.caption("🚀 Powered by Streamlit • Plotly • AWS Elastic Beanstalk")
