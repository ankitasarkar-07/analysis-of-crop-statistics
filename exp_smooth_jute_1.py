import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('wb_jute_production.csv')

# Group data by Crop_Year and calculate mean for weighted_mean_production
seasonal_df = df.groupby('Crop_Year')['weighted_mean_production'].mean().reset_index()

# Filter actual data up to 2020
actual_data = seasonal_df[seasonal_df['Crop_Year'] <= 2020]

# Initialize variables
alpha = 0.9  # Smoothing factor (higher weight for recent data)
beta = 0.2  # Trend smoothing factor (adjustable for trend behavior)
manual_smoothing = []  # To store smoothed values
trend = []  # To store trend values

# Perform Exponential Smoothing with trend component
for i, value in enumerate(actual_data['weighted_mean_production']):
    if i == 0:  # Initialize with the first data point
        manual_smoothing.append(value)
        trend.append(0)  # No trend for the first point
    else:  # Apply smoothing formula with trend adjustment
        smoothed_value = alpha * value + (1 - alpha) * (manual_smoothing[-1] + trend[-1])
        new_trend = beta * (smoothed_value - manual_smoothing[-1]) + (1 - beta) * trend[-1]
        manual_smoothing.append(smoothed_value)
        trend.append(new_trend)

# Forecast from 2021 to 2025 using the trend
forecast_years = list(range(2021, 2026))
forecast = []
last_smoothed = manual_smoothing[-1]
last_trend = trend[-1]
for year in forecast_years:
    forecast_value = last_smoothed + last_trend  # Add the trend to the last smoothed value
    forecast.append(forecast_value)
    last_smoothed = forecast_value  # Update for next step
    last_trend = last_trend  # Keep the trend constant

# Combine years and values for plotting
actual_years = actual_data['Crop_Year']
forecast_values = [None] * len(actual_years) + forecast
all_years = list(actual_years) + forecast_years

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual_years, actual_data['weighted_mean_production'], label='Historical Data', color='blue')
plt.plot(forecast_years, forecast, label='Forecast (2021-2025)', color='orange', linestyle='--', marker='o')
plt.title('Jute Production Forecast (using Exponential Smoothing)')
plt.xlabel('Year')
plt.ylabel('Weighted Mean Production')
plt.legend()
#plt.grid()
plt.show()
import numpy as np

# Actual values (replace with your actual test data)
actual_values = [105, 110, 120, 125, 130]

# Forecasted values (replace with your model's forecasted data)
forecasted_values = [100, 112, 118, 127, 135]

# Ensure both arrays are NumPy arrays for easy computation
actual_values = np.array(actual_values)
forecasted_values = np.array(forecasted_values)

# Calculate MAE
mae = np.mean(np.abs(actual_values - forecasted_values))

# Calculate RMSE
rmse = np.sqrt(np.mean(np.square(actual_values - forecasted_values)))

# Calculate MAPE (exclude zero actual values to avoid NaN)
mape = np.mean(np.abs((actual_values - forecasted_values) / actual_values)) * 100

# Print the results
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

