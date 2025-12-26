# Multilayer Perceptron (MLP) for Iris classification
# with loss and accuracy plotted on the same graph

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("iris.csv")

# Separate features and labels
X = data.drop("species", axis=1)
y = data["species"]

# Encode class labels (string -> integer)
y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model and store history
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test Loss", loss)

# Plot loss and accuracy on the same graph
plt.plot(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(["Loss", "Accuracy"])
plt.title("Training Loss and Accuracy")
plt.show()
