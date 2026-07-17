import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

# ---------------- Load Model ----------------

from xgboost import XGBRegressor

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("xgb_model.json")
    return model

model = load_model()

# ---------------- Sidebar ----------------

st.sidebar.title("🛒 Retail Forecasting")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Sales Prediction",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)

# ---------------- Home ----------------

if menu == "🏠 Home":

    st.header("📌 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "M5 Forecasting")

    with col2:
        st.metric("Models", "4")

    with col3:
        st.metric("Final Model", "XGBoost")

    st.markdown("---")

    st.subheader("📖 About the Project")

    st.write("""
This project predicts future retail product demand using Machine Learning
and Deep Learning techniques. It helps improve inventory planning and
decision-making in retail businesses.
""")

    st.subheader("🛠 Models Used")

    st.markdown("""
- XGBoost
- CatBoost
- LightGBM
- LSTM
""")

    st.success("🏆 Final Selected Model : XGBoost")

# ---------------- Sales Prediction ----------------

elif menu == "📈 Sales Prediction":

    st.header("📈 Sales Prediction")
    st.write("Enter the product details below and click Predict.")

    col1, col2 = st.columns(2)

    with col1:
        wm_yr_wk = st.number_input("Week Number", value=11101)
        wday = st.number_input("Week Day", min_value=1, max_value=7, value=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        year = st.number_input("Year", value=2016)
        sell_price = st.number_input("Sell Price", min_value=0.0, value=5.0)

    with col2:
        lag_28 = st.number_input("Lag 28 Sales", min_value=0.0, value=0.0)
        lag_35 = st.number_input("Lag 35 Sales", min_value=0.0, value=0.0)
        lag_42 = st.number_input("Lag 42 Sales", min_value=0.0, value=0.0)
        lag_56 = st.number_input("Lag 56 Sales", min_value=0.0, value=0.0)

    if st.button("🔮 Predict Sales"):

        input_data = pd.DataFrame([{
            "wm_yr_wk": wm_yr_wk,
            "wday": wday,
            "month": month,
            "year": year,
            "event_name_1": 0,
            "event_type_1": 0,
            "event_name_2": 0,
            "event_type_2": 0,
            "sell_price": sell_price,
            "price_missing_flag": 0,
            "day_of_week": wday,
            "day_of_month": 15,
            "week_of_year": 20,
            "is_weekend": 0,
            "is_month_start": 0,
            "is_month_end": 0,
            "lag_28": lag_28,
            "lag_35": lag_35,
            "lag_42": lag_42,
            "lag_56": lag_56,
            "rolling_mean_7": lag_28,
            "rolling_mean_14": lag_35,
            "rolling_mean_28": lag_42,
            "rolling_std_7": 0,
            "price_change_pct": 0,
            "dept_avg_price": sell_price,
            "price_vs_dept_avg": 1,
            "store_avg_price": sell_price,
            "price_vs_store_avg": 1,
            "snap": 0,
            "has_event": 0,
            "days_to_christmas": 200,
            "item_id_enc": 0,
            "dept_id_enc": 0,
            "cat_id_enc": 0,
            "store_id_enc": 0,
            "state_id_enc": 0
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"✅ Predicted Sales : {prediction:.2f} Units")

# ---------------- Model Performance ----------------

elif menu == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.table(
        pd.DataFrame({
            "Model": ["XGBoost", "CatBoost", "LightGBM", "LSTM"],
            "RMSE": [2.1631, 2.1998, 2.2377, 2.3289],
            "MAE": [1.1389, 1.1572, 1.1765, 1.1456],
            "MAPE (%)": [54.60, 53.49, 51.33, 67.39]
        })
    )

# ---------------- About ----------------

elif menu == "ℹ️ About":

    st.header("ℹ️ About")

    st.write("""
**Project Title**

Enterprise Retail Intelligence Platform for Demand Forecasting,
Inventory Optimization and Decision Support.

**Tools Used**

- Python
- Streamlit
- XGBoost
- CatBoost
- LightGBM
- LSTM
- Pandas
- NumPy
""")