# ===== Imports =====
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ===== Load Dataset (MNIST) =====
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ===== Normalize & Reshape =====
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension: (28,28) → (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

print("Train:", x_train.shape, y_train_cat.shape)
print("Test :", x_test.shape, y_test_cat.shape)

# ===== Build CNN Model =====
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# ===== Compile Model =====
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===== Train Model =====
history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ===== Evaluate Model =====
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print("Test loss :", test_loss)
print("Test accuracy :", test_acc)

# ===== Predictions =====
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ===== Accuracy Graph =====
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ===== Loss Graph =====
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()