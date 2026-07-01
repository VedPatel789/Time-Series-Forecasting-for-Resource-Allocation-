import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

sns.set(style="darkgrid")

# ==========================================================
# 1. GENERATE SYNTHETIC REALISTIC DEMAND DATA
# ==========================================================

np.random.seed(42)

n_days = 500

dates = pd.date_range(start="2023-01-01", periods=n_days)

trend = np.linspace(50, 200, n_days)
seasonality = 20 * np.sin(np.linspace(0, 25, n_days))
noise = np.random.normal(0, 8, n_days)

demand = trend + seasonality + noise

df = pd.DataFrame({
    "Date": dates,
    "Demand": demand
})

df.set_index("Date", inplace=True)

print("\nDataset Created Successfully")
print(df.head())
# ==========================================================
# 2. TRAIN / TEST SPLIT
# ==========================================================

train_size = int(len(df) * 0.8)

train, test = df.iloc[:train_size], df.iloc[train_size:]

print("\nTrain size:", len(train))
print("Test size:", len(test))

# ==========================================================
# 3. ARIMA MODEL TRAINING
# ==========================================================

print("\nTraining ARIMA model...")

model = ARIMA(train["Demand"], order=(5, 1, 0))
model_fit = model.fit()

print("Model trained successfully")

# ==========================================================
# 4. FORECASTING
# ==========================================================

forecast_steps = len(test)

forecast = model_fit.forecast(steps=forecast_steps)

forecast = pd.Series(forecast, index=test.index)

print("\nForecast generated")
print(forecast.head())

# ==========================================================
# 5. PLOT: ACTUAL vs FORECAST
# ==========================================================

plt.figure(figsize=(14,6))

plt.plot(train.index, train["Demand"], label="Train Data", color="blue")
plt.plot(test.index, test["Demand"], label="Test Data", color="green")
plt.plot(forecast.index, forecast, label="Forecast", color="red")

plt.title("Actual vs Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Demand")

plt.legend()

plt.show()
# ==========================================================
# 6. EVALUATION METRICS
# ==========================================================

mae = mean_absolute_error(test["Demand"], forecast)
rmse = np.sqrt(mean_squared_error(test["Demand"], forecast))
mape = np.mean(np.abs((test["Demand"] - forecast) / test["Demand"])) * 100

print("\n================= MODEL PERFORMANCE =================")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("=====================================================\n")

# ==========================================================
# 7. RESOURCE ALLOCATION LOGIC
# ==========================================================

allocation = []

for value in forecast:
    if value < 80:
        allocation.append("LOW (Few resources needed)")
    elif value < 140:
        allocation.append("MEDIUM (Moderate resources)")
    else:
        allocation.append("HIGH (Scale up resources)")

allocation = pd.Series(allocation, index=forecast.index)

result_df = pd.DataFrame({
    "Actual": test["Demand"],
    "Forecast": forecast,
    "Allocation": allocation
})

print("\nResource Allocation Sample:")
print(result_df.head())

# ==========================================================
# 8. PEAK DEMAND ALERTS
# ==========================================================

threshold = df["Demand"].mean() + df["Demand"].std()

alerts = forecast[forecast > threshold]

print("\n================= PEAK DEMAND ALERTS =================")
print(f"Threshold: {threshold:.2f}")

if len(alerts) == 0:
    print("No peak demand days detected.")
else:
    print(f"High demand detected on {len(alerts)} days.")
    print(alerts)
print("======================================================\n")
# ==========================================================
# 9. FINAL VISUALIZATION (CLEAN DASHBOARD STYLE)
# ==========================================================

plt.figure(figsize=(14,6))

plt.plot(df.index, df["Demand"], label="Full Data", alpha=0.5)
plt.plot(forecast.index, forecast, label="Forecast", color="red", linewidth=2)

plt.title("Demand Forecast Overview")
plt.xlabel("Date")
plt.ylabel("Demand")

plt.legend()
plt.show()

# ==========================================================
# 10. FORECAST + ACTUAL COMPARISON PLOT
# ==========================================================

plt.figure(figsize=(14,6))

plt.plot(test.index, test["Demand"], label="Actual Demand", color="green")
plt.plot(forecast.index, forecast, label="Predicted Demand", color="red")

plt.title("Actual vs Predicted Demand")
plt.xlabel("Date")
plt.ylabel("Demand")

plt.legend()
plt.show()

# ==========================================================
# 11. EXPORT RESULTS TO CSV
# ==========================================================

final_output = pd.DataFrame({
    "Actual_Demand": test["Demand"],
    "Forecast_Demand": forecast,
    "Resource_Allocation": allocation
})

final_output.to_csv("resource_allocation_results.csv")

print("\nCSV file saved: resource_allocation_results.csv")

# ==========================================================
# 12. EXECUTIVE SUMMARY
# ==========================================================

print("\n================= EXECUTIVE SUMMARY =================")

print("1. The dataset shows a clear trend + seasonality pattern.")
print("2. ARIMA model was used for time-series forecasting.")
print("3. Model performance metrics indicate prediction accuracy:")
print(f"   - MAE: {mae:.2f}")
print(f"   - RMSE: {rmse:.2f}")
print(f"   - MAPE: {mape:.2f}%")

print("\n4. Resource allocation strategy:")
print("   - LOW demand → minimal resources")
print("   - MEDIUM demand → moderate scaling")
print("   - HIGH demand → infrastructure scaling required")

print("\n5. Peak demand alerts were generated for risk management.")
