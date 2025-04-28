import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm

st.title("Time Series Analyzer")

uploaded_file = st.file_uploader("Upload your time series data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.subheader("Uploaded Data")
    st.dataframe(df.head())
else:
    st.info("Please upload a time series data file to proceed.")
    st.stop()

date_column = st.selectbox("Select the date column:", df.columns)
value_column = st.selectbox("Select the value column:", [col for col in df.columns if col != date_column])

try:
    df[date_column] = pd.to_datetime(df[date_column])
except Exception as e:
    st.error(f"Error converting date column: {e}. Please ensure it's in a recognizable date format.")
    st.stop()

df.sort_values(by=date_column, inplace=True)
df.reset_index(drop=True, inplace=True)
series = df[value_column]

# Time Series Plot
st.subheader("Time Series Plot")
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(series)+1), series)
plt.xlabel("Time Step")
plt.ylabel(value_column)
plt.title("Time Series Data")
st.pyplot(plt)

# Decomposition
st.subheader("Decomposition Plot")
model = st.selectbox("Select the type of decomposition:", ["Additive", "Multiplicative"])
seasons = st.number_input("Enter number of seasons per cycle", step=1, value=1)
decomposition = seasonal_decompose(series, model=model.lower(), period=seasons)
st.pyplot(decomposition.plot())

# ACF and PACF
st.subheader("ACF Plot")
fig_acf = plot_acf(series)
st.pyplot(fig_acf)

st.subheader("PACF Plot")
fig_pacf = plot_pacf(series)
st.pyplot(fig_pacf)

# ADF Test
st.subheader("ADF Test")
st.write("Null Hypothesis: The time series is non-stationary")
st.write("Alternative Hypothesis: The time series is stationary")
adf = adfuller(series)
st.write(f"Test Statistic: {adf[0]}")
st.write(f"P-value: {adf[1]}")
if adf[1] < 0.05:
    st.write("Since the p value is less than the level of significance at 5% level of significance, we reject the null hypothesis")
    st.write("So, the time series is STATIONARY")
else:
    st.write("Since the p value is greater than the level of significance at 5% level of significance, we accept the null hypothesis")
    st.write("So, the time series is NON-STATIONARY")
# AutoARIMA Forecast
st.subheader("AutoARIMA Forecast")
n_periods = st.number_input("Select number of periods to forecast:", min_value=1, value=12, step=1)
with st.spinner("Fitting AutoARIMA model..."):
    model_arima = pm.auto_arima(series, seasonal=True, m=seasons,
                                stepwise=True, suppress_warnings=True, error_action='ignore')
    st.success("AutoARIMA model fitted!")

st.text("Model Summary:")
st.text(model_arima.summary())

# Forecast
forecast = model_arima.predict(n_periods=n_periods)
forecast_index = [f"t+{i+1}" for i in range(n_periods)]
#forecast_df = pd.DataFrame({value_column: forecast}, index=forecast_index)

# Forecast Plot
st.subheader("Forecast Plot")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, len(series)+1), series, label='Historical')
ax.plot(range(len(series)+1, len(series)+n_periods+1), forecast, label='Forecast', linestyle='--')
ax.set_xlabel("Time Step")
ax.set_ylabel(value_column)
ax.set_title("Time Series Forecast (AutoARIMA)")
ax.legend()
st.pyplot(fig)

# Forecast Table
#st.subheader("Forecasted Values")
#st.dataframe(forecast_df)
st.write(forecast)

# Exponential Smoothing
st.subheader("Exponential Smoothing Forecast")
method = st.selectbox("Choose Exponential Smoothing Method", ["Simple", "Holt", "Holt-Winters"])
forecast_periods = st.number_input("Select number of periods to forecast:", min_value=1, value=36, step=1)

# Train-test split
train_size = int(len(series) * 0.8)
train, test = series.iloc[:train_size], series.iloc[train_size:]

# Fit model
if method == "Simple":
    model = SimpleExpSmoothing(train).fit()
elif method == "Holt":
    model = Holt(train).fit()
else:
    seasonal_periods = st.number_input("Seasonal Periods", min_value=1, value=seasons)
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()

# Forecast
test_forecast = model.forecast(len(test))
full_forecast = model.forecast(forecast_periods)

# Evaluation
mae = mean_absolute_error(test, test_forecast)
mape = mean_absolute_percentage_error(test, test_forecast) * 100

st.subheader("Model Accuracy")
st.write(f"*MAE*: {mae:.2f}")
st.write(f"*MAPE*: {mape:.2f}%")

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, len(train)+1), train, label='Train')
ax.plot(range(len(train)+1, len(train)+len(test)+1), test, label='Test')
ax.plot(range(len(train)+1, len(train)+len(test)+1), test_forecast, label='Test Forecast', linestyle='--')
ax.plot(range(len(train)+len(test)+1, len(train)+len(test)+forecast_periods+1), full_forecast, 
        label='Future Forecast', linestyle='--', color='blue')
ax.set_xlabel("Time Step")
ax.set_ylabel(value_column)
ax.set_title("Exponential Smoothing Forecast")
ax.legend()
st.pyplot(fig)
