import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the data
df = pd.read_csv('C:/Users/anton/Downloads/wb_rice_production.csv')

# Prepare the time series
# Use 'Total_Weighted_Production' and group it by year if needed
df['Crop_Year'] = pd.to_datetime(df['Crop_Year'], format='%Y')  # Convert to datetime
df.set_index('Crop_Year', inplace=True)  # Set Crop_Year as the index
time_series = df['Weighted_Mean_Production']

# Plot the line diagram
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Weighted_Mean_Production'], label='Weighted Mean Production', color='blue', linewidth=2)
plt.title('Rice Production Over the Years')
plt.xlabel('Year')
plt.ylabel('Weighted Mean Production')
plt.legend()
plt.show()

# Compute ACF and PACF
lag_acf = acf(time_series, nlags=20)
lag_pacf = pacf(time_series, nlags=20, method='ols')

# Plot ACF and PACF
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(121)
plot_acf(time_series, lags=20, ax=plt.gca())
plt.title('ACF Plot')

# PACF plot
plt.subplot(122)
plot_pacf(time_series, lags=20, ax=plt.gca())
plt.title('PACF Plot')

plt.tight_layout()
plt.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(time_series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

from statsmodels.tsa.arima.model import ARIMA

# Fit the model
model = ARIMA(time_series, order=(1, 0, 4))
model_fit = model.fit()
plt.axhline(y=0,color='black',linestyle='--',linewidth=1.5)

# Summary of the model
print(model_fit.summary())

# Plot residuals to ensure there's no pattern left
residuals = model_fit.resid
residuals.plot()

from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series (use the 'Weighted_Mean_Production' column)
result = seasonal_decompose(df['Weighted_Mean_Production'], period=4)  # Assuming yearly seasonal data
result.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller

# Seasonal differencing (Lag = 3 for seasons)
df['Seasonal_Diff'] = df['Weighted_Mean_Production'].diff(periods=3)



# Apply ADF test to seasonal-differenced data
result = adfuller(df['Seasonal_Diff'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])


from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(time_series,seasonal_order=(1, 1, 4, 3))  # Seasonal (P, D, Q, S)
model_fit = sarima_model.fit(disp=False)

# Step 4: Predict future values
# Predict for the next 10 years (or desired number of steps)
forecast_steps = 10
pred = model_fit.get_forecast(steps=forecast_steps)
pred_index = pd.date_range(start=time_series.index[-1], periods=forecast_steps + 1, freq='Y')[1:]
forecast = pred.predicted_mean
forecast.index = pred_index

# Step 5: Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Actual', color='blue')  # Original time series
plt.plot(forecast, label='Forecast', color='red')  # Forecasted values
plt.fill_between(forecast.index, 
                 pred.conf_int()['lower Weighted_Mean_Production'], 
                 pred.conf_int()['upper Weighted_Mean_Production'], 
                 color='pink', alpha=0.3, label='Confidence Interval')

plt.title('SARIMA Model: Forecast vs Actual')
plt.xlabel('Year')
plt.ylabel('Weighted Mean Production')
plt.legend()
plt.show()

# Step 6: (Optional) Evaluate the model
print(model_fit.summary())

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Step 2: Fit the Exponential Smoothing Model (Holt-Winters)
# Additive trend and seasonality, with seasonal period = 3 (Autumn, Summer, Winter)
model = ExponentialSmoothing(
    time_series,
    trend='add',
    seasonal='add',
    seasonal_periods=3
).fit()

# Step 3: Forecast Future Values
forecast_steps = 10  # Predict the next 10 seasons
forecast = model.forecast(steps=forecast_steps)

# Generate forecast index (aligned with future time periods)
forecast_index = pd.date_range(
    start=time_series.index[-1], periods=forecast_steps + 1, freq='Y'
)[1:]  # Skip the starting year as it's included in the last data point
forecast.index = forecast_index

# Step 4: Plot the Diagram
plt.figure(figsize=(12, 6))

# Plot the actual data (blue line)
plt.plot(time_series, label='Actual Data', color='blue')

# Plot the fitted values from the model (orange line)
plt.plot(model.fittedvalues, label='Fitted Values', color='orange')

# Plot the forecasted data (red line)
plt.plot(forecast.index, forecast, label='Forecasted Data', color='red')

# Add labels, title, and legend
plt.title('Exponential Smoothing: Actual vs Forecast')
plt.xlabel('Year')
plt.ylabel('Weighted Mean Production')
plt.legend()
plt.show()

pip install --upgrade pip setuptools wheel Cython
pip install numpy
pip install pmdarima
python -m venv arima_env
arima_env\Scripts\activate
