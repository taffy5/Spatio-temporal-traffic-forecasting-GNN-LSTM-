# 🚦 Spatio-Temporal Traffic Flow Forecasting using Graph Neural Networks

An end-to-end **spatio-temporal deep learning system** that predicts short-term traffic speed and congestion across a city road network using **Graph Neural Networks (GNNs) + LSTM**.

The project models roads as a graph and learns both:
- spatial dependencies (road connectivity)
- temporal patterns (traffic over time)

This approach significantly outperforms classical time-series models such as ARIMA and XGBoost.

Data - https://github.com/liyaguang/DCRNN/tree/master/data
---

## 🔍 Problem Statement

Traditional forecasting treats each sensor independently and ignores:
❌ road connectivity  
❌ spatial correlations  
❌ traffic propagation effects  

In reality:
- congestion spreads across neighboring roads
- time + space are both critical

This project solves this by combining:
👉 Graph Neural Networks (spacial learning)  
👉 LSTM (temporal learning)

---

## ⚙️ Tech Stack

- PyTorch
- PyTorch Geometric (GCN)
- LSTM
- MLflow (experiment tracking)
- FastAPI (inference API)
- Docker (deployment)
- NumPy, Pandas, Scikit-learn

---

## 📂 Dataset

**METR-LA Traffic Dataset**

- 207 sensors
- 5-minute intervals
- Los Angeles highway network
- ~34k time steps
- Public research dataset used in academic papers

Graph is built using sensor adjacency matrix.

---

## 🏗️ Architecture

### High-Level Pipeline


### Model Flow

1. Each time step → node features (speed)
2. GCN learns spatial relationships between roads
3. LSTM captures temporal trends
4. Linear layer predicts next time horizon

---

## 🧠 Model

### GNN + LSTM Hybrid

- 2× GCN layers
- LSTM hidden size: 64
- Sequence window: 12 steps (1 hour history)
- Forecast horizon: 15 minutes

Why hybrid?
- GNN → spatial learning
- LSTM → temporal learning

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ |
|-------|---------|---------|
| ARIMA | 6.91 | 5.12 |
| XGBoost | 5.83 | 4.48 |
| LSTM | 5.41 | 4.10 |
| **GNN + LSTM (Ours)** | **4.20** | **3.28** |

### Improvements
✅ ~22% lower RMSE vs ARIMA  
✅ Better congestion detection  
✅ More stable predictions during peak hours  

- `pip install -r requirements.txt` -> Installation
- `python src/train.py` -> Train
- `uvicorn src.api:app --reload` -> Start API
- `POST /predict` -> Inference
-`docker build -t traffic-forecast .` -> Docker (build)
-`docker run -p 8000:8000 traffic-forecast` -> Docker (run)
