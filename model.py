# 1. Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 2. Create training data
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
])

y = np.array([4, 5, 6, 7])

# 3. Reshape input for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))

# 5. Compile model
model.compile(optimizer='adam', loss='mse')

# 6. Train model
model.fit(X, y, epochs=300, verbose=0)

# 7. Predict next number
test_input = np.array([[5, 6, 7]])
test_input = test_input.reshape((1, 3, 1))

prediction = model.predict(test_input)
print("Predicted value:", prediction[0][0])
