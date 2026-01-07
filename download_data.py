# 0. Import Libraries
# -------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Dataset
# -------------------------
data = pd.read_csv("stock_dataset.csv")

# -------------------------
# 2. Clean & Prepare Data
# -------------------------
# Ensure Close column is numeric
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# Remove missing / invalid rows
data = data.dropna(subset=["Close"])

# Extract close prices
close_prices = data["Close"].values.reshape(-1, 1)

# -------------------------
# 3. Normalize Prices (0-1)
# -------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# -------------------------
# 4. Create Sequences
# -------------------------
sequence_length = 60  # use past 60 days to predict next day
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # past 60 days
    y.append(scaled_data[i])                    # next day

X = np.array(X)
y = np.array(y)

# -------------------------
# 5. Train-Test Split
# -------------------------
split = int(0.8 * len(X))  # 80% train, 20% test

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshape for LSTM (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# -------------------------
# 6. Build LSTM Model
# -------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))  # predicts the next closing price

model.compile(optimizer="adam", loss="mean_squared_error")

# -------------------------
# 7. Train Model
# -------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -------------------------
# 8. Predict
# -------------------------
predicted = model.predict(X_test)

# Convert back to original price scale
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------
# 9. Plot Results
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(real, label="Real Price")
plt.plot(predicted, label="Predicted Price")
plt.title("Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

