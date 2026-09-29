import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Retail Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 700;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBRegressor()

    model.load_model("xgb_model.json")

    return model


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

@st.cache_data
def load_validation_data():

    X_val = joblib.load("X_val.pkl")
    y_val = joblib.load("y_val.pkl")

    return X_val, y_val


# ============================================================
# LOAD RAW M5 DATA
# ============================================================

@st.cache_data
def load_raw_data():

    sales = pd.read_csv(
        "sales_train_validation.csv"
    )

    calendar = pd.read_csv(
        "calendar.csv"
    )

    prices = pd.read_csv(
        "sell_prices.csv"
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    return sales, calendar, prices


# ============================================================
# LOAD EVERYTHING
# ============================================================

model = load_model()

X_val, y_val = load_validation_data()

sales, calendar, prices = load_raw_data()

feature_names = list(X_val.columns)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Retail Intelligence")

st.sidebar.write(
    "Enterprise Retail Intelligence Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Sales Prediction",
        "📊 Sales Trend Analysis",
        "📦 Inventory Optimization",
        "🚨 Inventory Alert",
        "🔍 Explainable AI",
        "📊 Model Performance",
        "📥 Download Report",
        "ℹ️ About"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🏢 Enterprise Retail Intelligence Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Demand Forecasting, Inventory Optimization "
        "and Decision Support System"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📦 Products",
            "30,490"
        )

    with col2:

        st.metric(
            "🏪 Stores",
            "10"
        )

    with col3:

        st.metric(
            "📍 States",
            "3"
        )

    with col4:

        st.metric(
            "📊 Validation Rows",
            f"{len(X_val):,}"
        )

    st.divider()

    st.subheader(
        "🎯 Business Objectives"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 📈 Demand Forecasting

        - Predict future product demand
        - Identify demand patterns
        - Support sales planning
        """)

        st.markdown("""
        ### 📦 Inventory Optimization

        - Estimate safety stock
        - Calculate reorder point
        - Reduce stock-out risk
        """)

    with col2:

        st.markdown("""
        ### 🔍 Explainable AI

        - Identify important features
        - Understand model behaviour
        - Support business decisions
        """)

        st.markdown("""
        ### 🚨 Inventory Alerts

        - Compare current stock with reorder point
        - Generate inventory status
        - Support replenishment decisions
        """)


# ============================================================
# SALES PREDICTION
# ============================================================

elif page == "📈 Sales Prediction":

    st.title(
        "📈 Sales Prediction"
    )

    st.write(
        "Enter simple business information. "
        "Technical features such as lag and rolling "
        "features are calculated automatically."
    )

    st.divider()

    # ========================================================
    # PRODUCT
    # ========================================================

    product_list = sorted(
        sales["item_id"].unique().tolist()
    )

    selected_product = st.selectbox(
        "📦 Select Product",
        product_list
    )


    # ========================================================
    # STORE
    # ========================================================

    store_list = sorted(
        sales["store_id"].unique().tolist()
    )

    selected_store = st.selectbox(
        "🏪 Select Store",
        store_list
    )


    # ========================================================
    # AUTOMATIC STATE
    # ========================================================

    store_state_mapping = (
        sales[
            ["store_id", "state_id"]
        ]
        .drop_duplicates()
        .set_index("store_id")["state_id"]
        .to_dict()
    )

    automatic_state = store_state_mapping.get(
        selected_store,
        ""
    )


    st.info(
        f"📍 Store State: **{automatic_state}**"
    )


    # ========================================================
    # FORECAST DATE
    # ========================================================

    min_date = calendar["date"].min()
    max_date = calendar["date"].max()

    selected_date = st.date_input(
        "📅 Forecast Date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    selected_date = pd.to_datetime(
        selected_date
    )


    # ========================================================
    # SELLING PRICE
    # ========================================================

    price_value = st.number_input(
        "💰 Selling Price",
        min_value=0.0,
        value=1.0,
        step=0.01
    )


    # ========================================================
    # EVENT
    # ========================================================

    date_calendar_row = calendar[
        calendar["date"] == selected_date
    ]

    calendar_event = "No Event"

    if not date_calendar_row.empty:

        event_name = date_calendar_row.iloc[0].get(
            "event_name_1",
            np.nan
        )

        if pd.notna(event_name):

            calendar_event = str(
                event_name
            )


    event_options = [
        "No Event"
    ]

    if calendar_event != "No Event":

        event_options.append(
            calendar_event
        )

    selected_event = st.selectbox(
        "🎉 Event / Holiday",
        event_options
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Sales",
        type="primary"
    ):

        with st.spinner(
            "Preparing features and generating prediction..."
        ):

            # =================================================
            # GET PRODUCT + STORE
            # =================================================

            product_store = sales[
                (sales["item_id"] == selected_product) &
                (sales["store_id"] == selected_store)
            ].copy()


            if product_store.empty:

                st.error(
                    "Selected Product and Store combination "
                    "is not available."
                )

                st.stop()


            # =================================================
            # PRODUCT INFORMATION
            # =================================================

            selected_product_row = product_store.iloc[0]

            selected_dept = selected_product_row[
                "dept_id"
            ]

            selected_cat = selected_product_row[
                "cat_id"
            ]

            selected_state = selected_product_row[
                "state_id"
            ]


            # =================================================
            # GET TARGET DATE
            # =================================================

            date_row = calendar[
                calendar["date"] == selected_date
            ]


            if date_row.empty:

                st.error(
                    "Selected date is not available "
                    "in the calendar."
                )

                st.stop()


            date_info = date_row.iloc[0]

            target_d = date_info["d"]


            # =================================================
            # MELT PRODUCT SALES
            # =================================================

            id_columns = [
                "id",
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id"
            ]


            daily_sales = product_store.melt(
                id_vars=id_columns,
                var_name="d",
                value_name="sales"
            )


            # =================================================
            # MERGE CALENDAR
            # =================================================

            daily_sales = daily_sales.merge(
                calendar,
                on="d",
                how="left"
            )


            daily_sales = daily_sales.sort_values(
                "date"
            )


            # =================================================
            # PRICE DATA
            # =================================================

            price_data = prices[
                (prices["item_id"] == selected_product) &
                (prices["store_id"] == selected_store)
            ].copy()


            if not price_data.empty:

                daily_sales = daily_sales.merge(
                    price_data[
                        [
                            "store_id",
                            "item_id",
                            "wm_yr_wk",
                            "sell_price"
                        ]
                    ],
                    on=[
                        "store_id",
                        "item_id",
                        "wm_yr_wk"
                    ],
                    how="left"
                )

            else:

                daily_sales["sell_price"] = np.nan


            # =================================================
            # PRICE MISSING FLAG
            # =================================================

            daily_sales["price_missing_flag"] = (
                daily_sales["sell_price"]
                .isna()
                .astype(int)
            )


            # =================================================
            # USE HISTORICAL DATA ONLY
            # =================================================

            history = daily_sales[
                daily_sales["date"] < selected_date
            ].copy()


            if len(history) < 42:

                st.error(
                    "Not enough historical data to calculate "
                    "42-day lag features."
                )

                st.stop()


            # =================================================
            # LAG FEATURES
            # =================================================

            history["lag_1"] = (
                history["sales"]
                .shift(1)
            )

            history["lag_7"] = (
                history["sales"]
                .shift(7)
            )

            history["lag_14"] = (
                history["sales"]
                .shift(14)
            )

            history["lag_21"] = (
                history["sales"]
                .shift(21)
            )

            history["lag_28"] = (
                history["sales"]
                .shift(28)
            )

            history["lag_35"] = (
                history["sales"]
                .shift(35)
            )

            history["lag_42"] = (
                history["sales"]
                .shift(42)
            )


            # =================================================
            # PREVIOUS SALES
            # =================================================

            previous_sales = (
                history["sales"]
                .shift(1)
            )


            # =================================================
            # ROLLING MEAN
            # =================================================

            history["rolling_mean_7"] = (
                previous_sales
                .rolling(7)
                .mean()
            )

            history["rolling_mean_14"] = (
                previous_sales
                .rolling(14)
                .mean()
            )

            history["rolling_mean_28"] = (
                previous_sales
                .rolling(28)
                .mean()
            )


            # =================================================
            # ROLLING STD
            # =================================================

            history["rolling_std_7"] = (
                previous_sales
                .rolling(7)
                .std()
            )

            history["rolling_std_14"] = (
                previous_sales
                .rolling(14)
                .std()
            )

            history["rolling_std_28"] = (
                previous_sales
                .rolling(28)
                .std()
            )


            # =================================================
            # PRICE FEATURES
            # =================================================

            history["price_change"] = (
                history["sell_price"]
                .diff()
            )


            history["price_change_pct"] = (
                history["sell_price"]
                .pct_change()
            )


            history["price_rolling_mean_7"] = (
                history["sell_price"]
                .rolling(7)
                .mean()
            )


            history["price_rolling_mean_28"] = (
                history["sell_price"]
                .rolling(28)
                .mean()
            )


            # =================================================
            # LAST HISTORICAL ROW
            # =================================================

            latest = history.iloc[-1].copy()


            # =================================================
            # TARGET DATE FEATURES
            # =================================================

            target_year = selected_date.year

            target_month = selected_date.month

            target_week = int(
                selected_date.isocalendar().week
            )

            target_day = selected_date.day

            target_wday = (
                selected_date.weekday() + 1
            )


            # =================================================
            # TARGET PRICE
            # =================================================

            historical_price = latest[
                "sell_price"
            ]

            if pd.isna(historical_price):

                historical_price = price_value


            target_price = price_value


            # =================================================
            # PRICE CHANGE
            # =================================================

            target_price_change = (
                target_price -
                historical_price
            )


            if historical_price != 0:

                target_price_change_pct = (
                    target_price_change /
                    historical_price
                )

            else:

                target_price_change_pct = 0


            # =================================================
            # PRICE ROLLING MEAN
            # =================================================

            previous_prices = (
                history["sell_price"]
                .dropna()
            )


            if len(previous_prices) >= 7:

                price_mean_7 = (
                    previous_prices
                    .tail(7)
                    .mean()
                )

            else:

                price_mean_7 = target_price


            if len(previous_prices) >= 28:

                price_mean_28 = (
                    previous_prices
                    .tail(28)
                    .mean()
                )

            else:

                price_mean_28 = target_price


            # =================================================
            # EVENT INFORMATION
            # =================================================

            target_event_name_1 = (
                date_info.get(
                    "event_name_1",
                    np.nan
                )
            )

            target_event_type_1 = (
                date_info.get(
                    "event_type_1",
                    np.nan
                )
            )

            target_event_name_2 = (
                date_info.get(
                    "event_name_2",
                    np.nan
                )
            )

            target_event_type_2 = (
                date_info.get(
                    "event_type_2",
                    np.nan
                )
            )


            # User selected event override

            if selected_event == "No Event":

                target_event_name_1 = np.nan

                target_event_type_1 = np.nan

            elif selected_event != calendar_event:

                target_event_name_1 = selected_event


            # =================================================
            # SNAP FEATURES
            # =================================================

            snap_ca = date_info.get(
                "snap_CA",
                0
            )

            snap_tx = date_info.get(
                "snap_TX",
                0
            )

            snap_wi = date_info.get(
                "snap_WI",
                0
            )


            # =================================================
            # ENCODING
            # =================================================

            item_values = sorted(
                sales["item_id"]
                .dropna()
                .unique()
                .tolist()
            )

            dept_values = sorted(
                sales["dept_id"]
                .dropna()
                .unique()
                .tolist()
            )

            cat_values = sorted(
                sales["cat_id"]
                .dropna()
                .unique()
                .tolist()
            )

            store_values = sorted(
                sales["store_id"]
                .dropna()
                .unique()
                .tolist()
            )

            state_values = sorted(
                sales["state_id"]
                .dropna()
                .unique()
                .tolist()
            )


            item_mapping = {
                value: index
                for index, value
                in enumerate(item_values)
            }

            dept_mapping = {
                value: index
                for index, value
                in enumerate(dept_values)
            }

            cat_mapping = {
                value: index
                for index, value
                in enumerate(cat_values)
            }

            store_mapping = {
                value: index
                for index, value
                in enumerate(store_values)
            }

            state_mapping = {
                value: index
                for index, value
                in enumerate(state_values)
            }


            # =================================================
            # EVENT ENCODING
            # =================================================

            event_name_values = (
                calendar["event_name_1"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            event_name_values = sorted(
                event_name_values
            )


            event_type_values = (
                calendar["event_type_1"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            event_type_values = sorted(
                event_type_values
            )


            event_name_2_values = (
                calendar["event_name_2"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            event_name_2_values = sorted(
                event_name_2_values
            )


            event_type_2_values = (
                calendar["event_type_2"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            event_type_2_values = sorted(
                event_type_2_values
            )


            event_name_mapping = {
                value: index
                for index, value
                in enumerate(event_name_values)
            }

            event_type_mapping = {
                value: index
                for index, value
                in enumerate(event_type_values)
            }

            event_name_2_mapping = {
                value: index
                for index, value
                in enumerate(event_name_2_values)
            }

            event_type_2_mapping = {
                value: index
                for index, value
                in enumerate(event_type_2_values)
            }


            # =================================================
            # EVENT ENCODE FUNCTION
            # =================================================

            def encode_event(
                value,
                mapping
            ):

                if pd.isna(value):

                    return 0

                return mapping.get(
                    str(value),
                    0
                )


            # =================================================
            # BUILD INPUT
            # =================================================

            input_dict = {
                feature: 0
                for feature in feature_names
            }


            # =================================================
            # LAG VALUES
            # =================================================

            input_dict["lag_1"] = (
                latest["lag_1"]
            )

            input_dict["lag_7"] = (
                latest["lag_7"]
            )

            input_dict["lag_14"] = (
                latest["lag_14"]
            )

            input_dict["lag_21"] = (
                latest["lag_21"]
            )

            input_dict["lag_28"] = (
                latest["lag_28"]
            )

            input_dict["lag_35"] = (
                latest["lag_35"]
            )

            input_dict["lag_42"] = (
                latest["lag_42"]
            )


            # =================================================
            # ROLLING FEATURES
            # =================================================

            input_dict["rolling_mean_7"] = (
                latest["rolling_mean_7"]
            )

            input_dict["rolling_mean_14"] = (
                latest["rolling_mean_14"]
            )

            input_dict["rolling_mean_28"] = (
                latest["rolling_mean_28"]
            )

            input_dict["rolling_std_7"] = (
                latest["rolling_std_7"]
            )

            input_dict["rolling_std_14"] = (
                latest["rolling_std_14"]
            )

            input_dict["rolling_std_28"] = (
                latest["rolling_std_28"]
            )


            # =================================================
            # DATE FEATURES
            # =================================================

            input_dict["year"] = target_year

            input_dict["month"] = target_month

            input_dict["week_of_year"] = target_week

            input_dict["day_of_month"] = target_day

            input_dict["wday"] = target_wday


            # =================================================
            # EVENT FEATURES
            # =================================================

            input_dict["event_name_1"] = (
                encode_event(
                    target_event_name_1,
                    event_name_mapping
                )
            )

            input_dict["event_type_1"] = (
                encode_event(
                    target_event_type_1,
                    event_type_mapping
                )
            )

            input_dict["event_name_2"] = (
                encode_event(
                    target_event_name_2,
                    event_name_2_mapping
                )
            )

            input_dict["event_type_2"] = (
                encode_event(
                    target_event_type_2,
                    event_type_2_mapping
                )
            )


            # =================================================
            # SNAP FEATURES
            # =================================================

            input_dict["snap_CA"] = snap_ca

            input_dict["snap_TX"] = snap_tx

            input_dict["snap_WI"] = snap_wi


            # =================================================
            # PRICE FEATURES
            # =================================================

            input_dict["sell_price"] = target_price

            input_dict["price_change"] = (
                target_price_change
            )

            input_dict["price_change_pct"] = (
                target_price_change_pct
            )

            input_dict["price_rolling_mean_7"] = (
                price_mean_7
            )

            input_dict["price_rolling_mean_28"] = (
                price_mean_28
            )

            input_dict["price_missing_flag"] = 0


            # =================================================
            # ENCODED ID FEATURES
            # =================================================

            input_dict["item_id_enc"] = (
                item_mapping.get(
                    selected_product,
                    0
                )
            )

            input_dict["dept_id_enc"] = (
                dept_mapping.get(
                    selected_dept,
                    0
                )
            )

            input_dict["cat_id_enc"] = (
                cat_mapping.get(
                    selected_cat,
                    0
                )
            )

            input_dict["store_id_enc"] = (
                store_mapping.get(
                    selected_store,
                    0
                )
            )

            input_dict["state_id_enc"] = (
                state_mapping.get(
                    selected_state,
                    0
                )
            )


            # =================================================
            # FINAL INPUT DATAFRAME
            # =================================================

            input_data = pd.DataFrame(
                [input_dict]
            )


            # =================================================
            # EXACT FEATURE ORDER
            # =================================================

            input_data = input_data[
                feature_names
            ]


            # =================================================
            # HANDLE NAN / INF
            # =================================================

            input_data = input_data.replace(
                [np.inf, -np.inf],
                np.nan
            )

            input_data = input_data.fillna(0)


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                input_data
            )


            predicted_sales = float(
                prediction[0]
            )


            predicted_sales = max(
                0,
                predicted_sales
            )


        # =====================================================
        # DISPLAY RESULT
        # =====================================================

        st.success(
            "✅ Sales prediction completed!"
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📦 Predicted Sales",
                f"{predicted_sales:.2f} units"
            )

        with col2:

            st.metric(
                "💰 Selling Price",
                f"{price_value:.2f}"
            )

        with col3:

            st.metric(
                "🏪 Store",
                selected_store
            )


        st.subheader(
            "📋 Prediction Details"
        )

        result_df = pd.DataFrame({
            "Parameter": [
                "Product",
                "Store",
                "State",
                "Forecast Date",
                "Selling Price",
                "Event",
                "Predicted Sales"
            ],

            "Value": [
                selected_product,
                selected_store,
                selected_state,
                selected_date.strftime(
                    "%Y-%m-%d"
                ),
                f"{price_value:.2f}",
                selected_event,
                f"{predicted_sales:.2f} units"
            ]
        })


        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SALES TREND ANALYSIS
# ============================================================

elif page == "📊 Sales Trend Analysis":

    st.title(
        "📊 Sales Trend Analysis"
    )

    st.write(
        "Actual vs Predicted sales on validation data."
    )

    y_pred = model.predict(X_val)

    y_pred = np.maximum(
        y_pred,
        0
    )


    trend_df = pd.DataFrame({

        "Actual Sales": np.asarray(
            y_val
        ),

        "Predicted Sales": np.asarray(
            y_pred
        )

    })


    sample_size = min(
        500,
        len(trend_df)
    )

    trend_sample = (
        trend_df
        .head(sample_size)
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        trend_sample[
            "Actual Sales"
        ],
        label="Actual"
    )

    ax.plot(
        trend_sample[
            "Predicted Sales"
        ],
        label="Predicted"
    )

    ax.set_title(
        "Actual vs Predicted Sales"
    )

    ax.set_xlabel(
        "Validation Samples"
    )

    ax.set_ylabel(
        "Sales"
    )

    ax.legend()

    st.pyplot(fig)


    st.subheader(
        "📋 Validation Sample"
    )

    display_df = trend_df.head(20).copy()

    display_df["Error"] = (
        display_df["Actual Sales"]
        -
        display_df["Predicted Sales"]
    )

    display_df["Absolute Error"] = (
        display_df["Error"]
        .abs()
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )


# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

elif page == "📦 Inventory Optimization":

    st.title(
        "📦 Inventory Optimization"
    )

    st.write(
        "Safety Stock and Reorder Point calculation "
        "based on validation forecast error."
    )

    # --------------------------------------------------------
    # GLOBAL INVENTORY VALUES
    # --------------------------------------------------------

    error_std = 2.1630792405167614

    average_daily_demand = 1.498793515438317

    service_level = 0.95

    z_value = 1.645

    lead_time = 7


    # --------------------------------------------------------
    # CALCULATE SAFETY STOCK
    # --------------------------------------------------------

    safety_stock = (
        z_value
        * error_std
        * np.sqrt(lead_time)
    )


    # --------------------------------------------------------
    # REORDER POINT
    # --------------------------------------------------------

    reorder_point = (
        average_daily_demand
        * lead_time
        + safety_stock
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Service Level",
            "95%"
        )

    with col2:

        st.metric(
            "⏱ Lead Time",
            "7 Days"
        )

    with col3:

        st.metric(
            "🛡 Safety Stock",
            f"{safety_stock:.2f}"
        )

    with col4:

        st.metric(
            "🔄 Reorder Point",
            f"{reorder_point:.2f}"
        )


    st.divider()


    st.subheader(
        "🧮 Inventory Calculation"
    )


    st.latex(
        r"""
        Safety\ Stock =
        Z \times Error\ Std \times \sqrt{Lead\ Time}
        """
    )


    st.latex(
        r"""
        Reorder\ Point =
        Average\ Daily\ Demand \times Lead\ Time
        + Safety\ Stock
        """
    )


    inventory_df = pd.DataFrame({

        "Metric": [
            "Average Daily Demand",
            "Forecast Error Std",
            "Service Level",
            "Z Value",
            "Lead Time",
            "Safety Stock",
            "Reorder Point"
        ],

        "Value": [
            average_daily_demand,
            error_std,
            service_level,
            z_value,
            lead_time,
            safety_stock,
            reorder_point
        ]

    })


    st.dataframe(
        inventory_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INVENTORY ALERT
# ============================================================

elif page == "🚨 Inventory Alert":

    st.title(
        "🚨 Inventory Alert"
    )

    st.write(
        "Check whether current inventory is below "
        "the calculated reorder point."
    )


    current_stock = st.number_input(
        "📦 Current Stock",
        min_value=0.0,
        value=20.0,
        step=1.0
    )


    reorder_point = (
        19.90583982466635
    )


    if st.button(
        "🔍 Check Inventory"
    ):

        if current_stock < reorder_point:

            st.error(
                "🚨 Reorder Required"
            )

            st.write(
                f"Current stock: "
                f"**{current_stock:.2f} units**"
            )

            st.write(
                f"Reorder point: "
                f"**{reorder_point:.2f} units**"
            )

            required_quantity = (
                reorder_point
                -
                current_stock
            )

            st.warning(
                f"Approximately "
                f"**{required_quantity:.2f} units** "
                f"are required to reach the reorder point."
            )

        else:

            st.success(
                "✅ Inventory Level is Healthy"
            )

            st.write(
                f"Current stock: "
                f"**{current_stock:.2f} units**"
            )

            st.write(
                f"Reorder point: "
                f"**{reorder_point:.2f} units**"
            )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔍 Explainable AI":

    st.title(
        "🔍 Explainable AI"
    )

    st.write(
        "Top features influencing the XGBoost demand "
        "forecasting model."
    )


    feature_importance = {

        "rolling_mean_28": 0.324293,

        "rolling_mean_14": 0.275100,

        "rolling_mean_7": 0.158779,

        "lag_28": 0.025688,

        "lag_35": 0.022883,

        "year": 0.018201,

        "wday": 0.015976,

        "price_missing_flag": 0.013617,

        "item_id_enc": 0.012519,

        "lag_42": 0.010874

    }


    importance_df = pd.DataFrame({

        "Feature":
            list(
                feature_importance.keys()
            ),

        "Importance":
            list(
                feature_importance.values()
            )

    })


    importance_df[
        "Importance %"
    ] = (
        importance_df["Importance"]
        * 100
    )


    st.subheader(
        "📊 Top Feature Importance"
    )


    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.barh(
        importance_df[
            "Feature"
        ][::-1],

        importance_df[
            "Importance %"
        ][::-1]
    )


    ax.set_xlabel(
        "Importance (%)"
    )

    ax.set_title(
        "Top 10 Features"
    )


    st.pyplot(fig)


    st.info(
        "Rolling demand features have high model "
        "importance because recent historical demand "
        "helps the model understand future demand patterns."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Validation performance comparison."
    )


    performance_df = pd.DataFrame({

        "Model": [
            "XGBoost",
            "CatBoost",
            "LightGBM",
            "LSTM"
        ],

        "RMSE": [
            2.1631,
            2.1998,
            2.2377,
            2.3289
        ],

        "MAE": [
            1.1389,
            1.1572,
            1.1765,
            1.1456
        ],

        "MAPE (%)": [
            54.60,
            53.49,
            51.33,
            67.39
        ]

    })


    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # XGBOOST ACTUAL VALIDATION METRICS
    # ========================================================

    y_pred = model.predict(
        X_val
    )

    y_pred = np.maximum(
        y_pred,
        0
    )


    mae = mean_absolute_error(
        y_val,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            y_pred
        )
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "XGBoost MAE",
            f"{mae:.4f}"
        )

    with col2:

        st.metric(
            "XGBoost RMSE",
            f"{rmse:.4f}"
        )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

elif page == "📥 Download Report":

    st.title(
        "📥 Download Validation Report"
    )

    st.write(
        "Download actual vs predicted validation results."
    )


    y_pred = model.predict(
        X_val
    )

    y_pred = np.maximum(
        y_pred,
        0
    )


    report_df = pd.DataFrame({

        "Actual Sales":
            np.asarray(y_val),

        "Predicted Sales":
            np.asarray(y_pred)

    })


    report_df["Error"] = (
        report_df["Actual Sales"]
        -
        report_df["Predicted Sales"]
    )


    report_df["Absolute Error"] = (
        report_df["Error"]
        .abs()
    )


    st.dataframe(
        report_df.head(100),
        use_container_width=True
    )


    csv_data = report_df.to_csv(
        index=False
    )


    st.download_button(

        label="📥 Download CSV Report",

        data=csv_data,

        file_name="sales_forecasting_validation_report.csv",

        mime="text/csv"

    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.write(
        "Enterprise Retail Intelligence Platform"
    )


    st.subheader(
        "🎯 Project Goal"
    )

    st.write(
        "The objective of this project is to build an "
        "intelligent retail decision-support system for "
        "demand forecasting, inventory optimization and "
        "explainable business insights."
    )


    st.subheader(
        "📊 Dataset"
    )

    st.write(
        "M5 Forecasting Competition Dataset"
    )


    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        "XGBoost is used for demand forecasting."
    )


    st.subheader(
        "📦 Inventory Optimization"
    )

    st.write(
        "Safety Stock and Reorder Point are calculated "
        "using forecast error and demand information."
    )


    st.subheader(
        "🔍 Explainable AI"
    )

    st.write(
        "Feature importance is used to understand "
        "which features contribute to model predictions."
    )


    st.subheader(
        "🚀 Deployment"
    )

    st.write(
        "The complete application is deployed using "
        "Streamlit."
    )