#  Time-Series Forecasting for Resource Allocation

## Overview
This project demonstrates an end-to-end **time-series forecasting system** using ARIMA to predict future demand and allocate resources efficiently.

It simulates real-world business scenarios such as:
- Cloud resource scaling
- Hospital staffing
- Inventory management
- Supply chain planning

---

## Problem Statement
Organizations often struggle with:
- Uncertain future demand
- Resource over-allocation (waste)
- Resource under-allocation (service failure)

This project solves it using predictive analytics.

---

## Features
- Synthetic demand generation (trend + seasonality)
- ARIMA time-series forecasting
- Train/test evaluation
- Error metrics (MAE, RMSE, MAPE)
- Resource allocation system (LOW / MEDIUM / HIGH)
- Peak demand detection
- Data visualization
- CSV report generation

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels (ARIMA)
- Scikit-learn

---

## Output
- Forecast graphs
- Actual vs predicted comparison
- Resource allocation table
- Performance metrics
- CSV report

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
