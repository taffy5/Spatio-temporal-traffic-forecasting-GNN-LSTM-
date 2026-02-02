# 🚦 Spatio-Temporal Traffic Flow Forecasting using Graph Neural Networks

An end-to-end **spatio-temporal deep learning system** that predicts short-term traffic speed and congestion across a city road network using **Graph Neural Networks (GNNs) + LSTM**.

The project models roads as a graph and learns both:
- spatial dependencies (road connectivity)
- temporal patterns (traffic over time)

This approach significantly outperforms classical time-series models such as ARIMA and XGBoost.

---

## 🔍 Problem Statement

Traditional forecasting treats each sensor independently and ignores:
❌ road connectivity  
❌ spatial correlations  
❌ traffic propagation effec
