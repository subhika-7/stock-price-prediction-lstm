
📈 SVMD-LSTM Stock Price Prediction

A hybrid financial time-series forecasting model combining Successive Variational Mode Decomposition (SVMD) and Long Short-Term Memory (LSTM) networks for accurate stock market predictions.

🔍 Overview

Stock price prediction is complex due to nonlinear, nonstationary, and volatile patterns. This project implements the SVMD-LSTM hybrid model:

SVMD decomposes stock price series into Intrinsic Mode Functions (IMFs) to reduce nonstationarity.

LSTM models are trained on each IMF to capture long-term dependencies.

Predicted IMFs are reconstructed to form the final forecast.

Datasets Used

SENSEX (India)

HSI – Hong Kong Hang Seng Index

S&P500 – US Index

WTI Crude Oil Prices

⚙️ Features

Reduces nonstationarity for better prediction.

Captures long-term trends in volatile financial time series.

Outperforms single models (LSTM, MLP, SVR) and hybrid models (VMD-LSTM, EMD-LSTM).

One-step-ahead forecasting using sliding window of 5 days.

Evaluation metrics: RMSE, MAE, MAPE, R², CID.

🚀 Installation
Clone the repository:
git clone https://github.com/yourusername/svmd-lstm-stock-prediction.git
cd svmd-lstm-stock-prediction

Install dependencies:
pip install -r requirements.txt


requirements.txt should include:

numpy
pandas
scikit-learn
tensorflow
matplotlib

📊 Results

The SVMD-LSTM model demonstrates improved accuracy over other models:

Dataset	RMSE	MAE	MAPE
HSI	52.28	41.84	0.162
SENSEX	158.91	122.67	0.214
S&P500	3.76	3.08	0.127
WTI	0.33	0.27	0.262

SVMD-LSTM > VMD-LSTM > EMD-LSTM > LSTM/MLP/SVR in accuracy.

Handles nonstationary and nonlinear patterns effectively.

🔮 Future Scope

Include additional features: trading volume, high/low prices.

Multi-step forecasting.

Advanced deep learning models and hyperparameter tuning (grid search, early stopping, dropout).

Extend to other financial or time-series forecasting problems.
