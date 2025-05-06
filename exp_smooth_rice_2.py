import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and clean the dataset
df = pd.read_csv("wb_rice_production.csv")
df['Season'] = df['Season'].str.strip()  # Remove leading/trailing spaces

# Set up storage for plots and metrics
forecast_results = {}
metrics = {}

# Define color map for plotting
season_colors = {
    'Autumn': 'green',
    'Summer': 'orange',
    'Winter': 'blue'
}

# Plotting setup
plt.figure(figsize=(14, 7))
plt.title('Exponential Smoothing Forecast: Rice Production by Season (2020–2022)')
plt.xlabel('Year')
plt.ylabel('Weighted Mean Production')

# Loop through each season
for season in ['Autumn', 'Summer', 'Winter']:
    # Filter data
    season_df = df[df['Season'] == season].copy()
    season_df = season_df.sort_values(by='Crop_Year')
    season_df.set_index('Crop_Year', inplace=True)

    # Extract target variable
    y = season_df['Weighted_Mean_Production']

    # Fit Exponential Smoothing model
    model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method='estimated')
    fit = model.fit()

    # Forecast next 3 years
    forecast = fit.forecast(3)
    forecast.index = [2020, 2021, 2022]

    # Store forecast
    forecast_results[season] = forecast

    # Plot historical and forecast separately
    plt.plot(y, label=f'{season} (Historical)', color=season_colors[season], linewidth=2)
    plt.plot(forecast, label=f'{season} (Forecast)', color=season_colors[season], linestyle='dashed', linewidth=2)

    # Evaluation on last 3 known years (2017–2019)
    y_true = y[-3:]
    y_pred = fit.fittedvalues[-3:]

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics[season] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Finalize plot
plt.legend()
plt.tight_layout()
plt.show()

# Print metrics
print("Model Evaluation (on 2017–2019 actuals):\n")
for season, metric in metrics.items():
    print(f"{season} Season:")
    print(f"  MAE : {metric['MAE']:.2f}")
    print(f"  RMSE: {metric['RMSE']:.2f}")
    print(f"  MAPE: {metric['MAPE']:.2f}%\n")
