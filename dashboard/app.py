import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(
    page_title="Global Market Analytics",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("Data/processed_data.csv")

# ======================
# SIDEBAR
# ======================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Home",
        "EDA",
        "Correlation",
        "Direction Prediction",
        "Volatility Prediction",
        "Anomaly Detection"
    ]
)

# ======================
# HOME
# ======================

if page == "Home":

    st.title("🌍 Global Market Analytics Dashboard")

    market = st.selectbox(
        "Select Market",
        df["Index"].unique()
    )

    market_df = df[df["Index"] == market]

    fig = px.line(
        market_df,
        x="Date",
        y="Close",
        title=f"{market} Closing Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Records")
    st.dataframe(market_df.tail(20))

# ======================
# EDA
# ======================

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write("Dataset Shape:", df.shape)

    st.write("Columns:")
    st.write(df.columns.tolist())

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

# ======================
# CORRELATION
# ======================

elif page == "Correlation":

    st.title("🔥 Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================
# DIRECTION PREDICTION
# ======================

elif page == "Direction Prediction":

    st.title("🤖 Market Direction Prediction")

    open_price = st.number_input("Open")
    high_price = st.number_input("High")
    low_price = st.number_input("Low")
    close_price = st.number_input("Close")
    volume = st.number_input("Volume")

    daily_return = st.number_input("Daily Return")
    ma20 = st.number_input("MA20")
    ma50 = st.number_input("MA50")
    volatility = st.number_input("Volatility")

    if st.button("Predict Direction"):

        model = joblib.load(
            "models/direction_model.pkl"
        )

        features = np.array([
            [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                daily_return,
                ma20,
                ma50,
                volatility
            ]
        ])

        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("📈 Market Likely UP")
        else:
            st.error("📉 Market Likely DOWN")

# ======================
# VOLATILITY
# ======================

elif page == "Volatility Prediction":

    st.title("⚡ Volatility Prediction")

    open_price = st.number_input("Open Price")
    high_price = st.number_input("High Price")
    low_price = st.number_input("Low Price")
    close_price = st.number_input("Close Price")
    volume = st.number_input("Volume")

    daily_return = st.number_input("Daily Return")
    ma20 = st.number_input("MA20")
    ma50 = st.number_input("MA50")

    if st.button("Predict Volatility"):

        model = joblib.load(
            "models/volatility_model.pkl"
        )

        features = np.array([
            [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                daily_return,
                ma20,
                ma50
            ]
        ])

        prediction = model.predict(features)[0]

        st.success(
            f"Predicted Volatility : {prediction:.4f}"
        )

# ======================
# ANOMALY DETECTION
# ======================

elif page == "Anomaly Detection":

    st.title("🚨 Anomaly Detection")

    model = joblib.load(
        "models/anomaly_model.pkl"
    )

    st.write(
        "Model Loaded Successfully"
    )

    st.success(
        "Anomaly Model Ready"
    )
    
    
