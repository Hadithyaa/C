
# ===================== Imports =====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks, optimizers

# ===================== Config =====================
DATA_FILE = r"C:\Users\postb\Downloads\CICIDS2017_sample.csv"
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===================== Load Data =====================
df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ===================== Label Column =====================
if "class" in df.columns:
    label_col = "class"
else:
    label_col = df.columns[-1]

print("Using label column:", label_col)

y_raw = df[label_col]

# ===================== Label Processing =====================
if pd.api.types.is_numeric_dtype(y_raw) and set(y_raw.unique()).issubset({0, 1}):
    y = y_raw.astype(int).values
else:
    y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).values

print("Label counts (0=normal, 1=attack):", np.bincount(y))

# ===================== Feature Processing =====================
X = df.drop(columns=[label_col]).copy()

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="ignore")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
X = X.values.astype(np.float32)

# ===================== Train-Test Split =====================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ===================== Scaling =====================
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ===================== Class Weights =====================
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {i: cw[i] for i in range(len(cw))}
print("Class weights:", class_weights)

# ===================== Model Definition =====================
def make_model(input_dim):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

model = make_model(x_train.shape[1])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===================== Training =====================
es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[es],
    verbose=1
)

# ===================== Evaluation =====================
y_prob = model.predict(x_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

test_auc = roc_auc_score(y_test, y_prob)
print("\nTest AUC:", test_auc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===================== ROC Curve =====================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

