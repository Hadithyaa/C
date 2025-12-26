# =========================
# IMPORTS
# =========================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# CONFIG
# =========================
SEQ_LEN = 20
FUTURE_STEPS = 30
TRAIN_POINTS = 500

# =========================
# DATA UTILITIES
# =========================
def generate_series(n=TRAIN_POINTS):
    # Trend + Sine + Noise (fake stock prices)
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 5 * np.sin(0.2 * t)
    noise = np.random.normal(0, 0.5, size=n)
    return (50 + trend + seasonal + noise).astype("float32")

def minmax_scale(x):
    x_min = x.min()
    x_max = x.max()
    scaled = (x - x_min) / (x_max - x_min + 1e-9)
    return scaled, x_min, x_max

def minmax_inverse(x_scaled, x_min, x_max):
    return x_scaled * (x_max - x_min + 1e-9) + x_min

def make_sequences(series, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X)[..., None]   # (samples, seq_len, 1)
    y = np.array(y)
    return X, y

# =========================
# BUILD LSTM MODEL
# =========================
def build_model(seq_len=SEQ_LEN):
    model = models.Sequential([
        layers.Input(shape=(seq_len, 1)),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# =========================
# GENERATE & PREPARE DATA
# =========================
history = generate_series()
scaled_history, x_min, x_max = minmax_scale(history)
X, y = make_sequences(scaled_history, SEQ_LEN)

# =========================
# MODEL TRAINING
# =========================
model = build_model(SEQ_LEN)
model.fit(X, y, epochs=25, batch_size=32, verbose=1)

# =========================
# FORECAST FUTURE POINTS
# =========================
last_seq = scaled_history[-SEQ_LEN:].copy()
future_scaled = []

for _ in range(FUTURE_STEPS):
    x_input = last_seq.reshape(1, SEQ_LEN, 1)
    pred = model.predict(x_input, verbose=0)[0, 0]
    future_scaled.append(pred)
    last_seq = np.concatenate([last_seq[1:], [pred]])

future = minmax_inverse(np.array(future_scaled), x_min, x_max)

# =========================
# PLOT WITH MATPLOTLIB
# =========================
past_x = np.arange(len(history))
future_x = np.arange(len(history), len(history) + FUTURE_STEPS)

plt.figure(figsize=(10, 4))
plt.plot(past_x, history, label="History")
plt.plot(future_x, future, "--", label="Forecast")
plt.title("Synthetic Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()