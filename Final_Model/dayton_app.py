import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO

# The file paths should ideally use os.path.join for cross-platform compatibility,
# but using the provided paths for now.
MODEL_PATH = "EBM_DAYTON.pkl"
FEATURE_COLS_PATH = "DAYTON_feature_cols.pkl"
DF_PATH = "DAYTON_df.pkl"
SCALER_PATH = "scaler_DAYTON.pkl"


@st.cache_resource
def load_model_and_artifacts():
    # 1. Load the EBM Model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. FIX: We swap the assignment here. 
    # The file "DAYTON_feature_cols.pkl" should contain the list of column names.
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f) # Correctly assigned to feature_cols

    # 3. Load the historical DataFrame
    df = pd.read_pickle(DF_PATH)

    # 4. FIX: We swap the assignment here. 
    # The file "scaler_DAYTON.pkl" should contain the StandardScaler object.
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f) # Correctly assigned to scaler

    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    
    # Return the artifacts in the correct order for destructuring
    return model, scaler, df, feature_cols


# Load all required artifacts
try:
    model, scaler, df, feature_cols = load_model_and_artifacts()
    LOAD_SUCCESS = True
except Exception as e:
    st.error(f"Error loading model or artifacts. Please ensure the file paths are correct and the files are not corrupted. Error: {e}")
    LOAD_SUCCESS = False
    
# Proceed only if loading was successful (necessary for a runnable script)
if LOAD_SUCCESS:
    
    def get_feature_importance(model, feature_cols):
        """Extracts feature importances from the EBM model if available."""
        try:
            # EBM models use 'feature_importances_'
            importances = model.feature_importances_
            s = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            return s
        except Exception:
            # Handle cases where the model might not have this attribute
            return None

    feature_importance = get_feature_importance(model, feature_cols)

    def make_features_for_next_hour(history_df, feature_cols):
        """Creates the feature row for the next hour based on the history."""
        last_index = history_df.index[-1]
        next_index = last_index + pd.Timedelta(hours=1)

        temp = history_df.copy()
        temp.loc[next_index, "load"] = np.nan # Placeholder for the target

        # 1. Time-based features
        temp["hour"] = temp.index.hour
        temp["dayofweek"] = temp.index.dayofweek
        temp["month"] = temp.index.month
        temp["dayofyear"] = temp.index.dayofyear

        # 2. Lag features
        temp["lag_1"] = temp["load"].shift(1)
        temp["lag_24"] = temp["load"].shift(24)
        temp["lag_168"] = temp["load"].shift(168)

        # 3. Rolling window features
        temp["roll_mean_24"] = temp["load"].shift(1).rolling(window=24).mean()
        temp["roll_std_24"] = temp["load"].shift(1).rolling(window=24).std()
        temp["roll_mean_168"] = temp["load"].shift(1).rolling(window=168).mean()
        temp["roll_std_168"] = temp["load"].shift(1).rolling(window=168).std()

        # 4. Fourier features (Daily)
        temp["fourier_day_sin"] = np.sin(2 * np.pi * temp.index.hour / 24)
        temp["fourier_day_cos"] = np.cos(2 * np.pi * temp.index.hour / 24)

        # 5. Fourier features (Yearly)
        t = np.arange(len(temp))
        period_year_hours = 365 * 24
        temp["fourier_year_sin"] = np.sin(2 * np.pi * t / period_year_hours)
        temp["fourier_year_cos"] = np.cos(2 * np.pi * t / period_year_hours)

        # Select the feature row for the next hour using the correct list of column names
        feat_row = temp.loc[[next_index], feature_cols]
        return feat_row


    def forecast_n_hours(n, df_base, model, scaler, feature_cols):
        """Performs iterative multi-step forecasting."""
        df_hist = df_base.copy()
        preds = []

        for _ in range(n):
            next_X = make_features_for_next_hour(df_hist, feature_cols)
            
            # Scale the features using the loaded scaler object
            next_X_scaled = pd.DataFrame(
                scaler.transform(next_X),
                index=next_X.index,
                columns=next_X.columns
            )
            
            # Predict
            y_hat = model.predict(next_X_scaled)[0]
            
            # Append the prediction to the history for next step's feature calculation (recursive forecasting)
            df_hist.loc[next_X.index[0], "load"] = y_hat
            preds.append((next_X.index[0], y_hat))

        return pd.DataFrame(preds, columns=["Datetime", "predicted_load"]).set_index("Datetime")


    def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Converts a DataFrame to CSV bytes for download."""
        return df.to_csv(index=True).encode("utf-8")


    # --- STREAMLIT UI ---
    st.set_page_config(page_title="DAYTON Load Forecast", layout="wide")

    st.title("⚡ DAYTON Hourly Load Forecast – EBM Prototype")

    st.write(
        """
    This app uses an **Explainable Boosting Machine (EBM)** trained on the 
    real **DAYTON hourly load** dataset to forecast future electricity demand (MW).

    - Historical data is loaded from your environment.
    - You can choose how many **future hours** to forecast.
    - You can download the forecast as CSV.
    - A small explanation panel shows the most important features.
    """
    )

    st.sidebar.header("Forecast Settings")
    n_hours = st.sidebar.slider(
        "Number of hours to forecast",
        min_value=1,
        max_value=168,
        value=24,
        step=1,
    )

    history_hours = st.sidebar.slider(
        "Show last N hours of history",
        min_value=24,
        max_value=24 * 14,
        value=24 * 7,
        step=24,
    )

    run_button = st.sidebar.button("Run Forecast")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recent Historical Load")
        st.line_chart(df["load"].iloc[-history_hours:])

    with col2:
        st.metric("Last observed time", df.index[-1].strftime("%Y-%m-%d %H:%M"))
        st.metric("Last observed load (MW)", f"{df['load'].iloc[-1]:.2f}")

    st.markdown("---")

    if run_button:
        st.subheader(f"Forecast for next {n_hours} hours")

        # The forecast function works correctly now because scaler and feature_cols are correct
        future_forecasts = forecast_n_hours(n_hours, df, model, scaler, feature_cols)

        left, right = st.columns([2, 1])

        with left:
            st.write("### Forecast Table")
            display_df = future_forecasts.reset_index().rename(
                columns={
                    "Datetime": "Timestamp",
                    "predicted_load": "Predicted Load (MW)",
                }
            )
            st.dataframe(display_df)

            csv_bytes = df_to_csv_bytes(display_df)
            st.download_button(
                label="⬇ Download forecast as CSV",
                data=csv_bytes,
                file_name="dayton_forecast.csv",
                mime="text/csv",
            )

        with right:
            st.write("### Model Explanation")

            if feature_importance is not None:
                st.write(
                    """
        The EBM model is **interpretable**.  
        Below are the most important features that drive the forecast:
        """
        )
                top_k = st.slider("Show top K features", 3, min(10, len(feature_importance)), 5)
                st.bar_chart(feature_importance.head(top_k))

                st.write("**Quick interpretation:**")
                st.markdown(
                    """
        - **Lag features** (`lag_24`, `lag_168`) capture the relationship to
          yesterday and last week at the same hour.
        - **Rolling means** (`roll_mean_24`, `roll_mean_168`) capture short-term and
          weekly trends or weather effects.
        - **Calendar features** (`hour`, `dayofweek`, `month`) capture daily and weekly
          usage patterns (evenings vs nights, weekdays vs weekends, seasons).
        - **Fourier terms** model smooth seasonal patterns that repeat daily or annually.
        """
        )
            else:
                st.info("Feature importance is not available for this model.")

        st.write("### History + Forecast")

        plot_history = df["load"].iloc[-history_hours:]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_history.index, plot_history.values, label="History")
        ax.plot(
            future_forecasts.index,
            future_forecasts["predicted_load"],
            label="Forecast",
        )
        ax.set_title("DAYTON Load – EBM Hourly Forecast")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Load (MW)")
        ax.legend()
        # Use st.line_chart for native Streamlit charting or st.pyplot for matplotlib
        # st.line_chart is usually better for interactivity, but we'll stick to st.pyplot as it was in the original code.
        st.pyplot(fig)

    else:
        st.info("⬅ Choose how many hours to forecast and click **Run Forecast**.")
        st.write("By default, you are seeing only the last few days of historical load.")
