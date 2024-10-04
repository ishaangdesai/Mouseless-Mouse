import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.keras.__version__)
# Load the datasets
right_click_data = pd.read_csv('rightClickTrainingData.csv')
left_click_data = pd.read_csv('leftClickTrainingData.csv')
general_data = pd.read_csv('generalTrainingData.csv')

# Display the first few rows of each dataset to understand the structure
print(right_click_data.head())
print(left_click_data.head())
print(general_data.head())

# Extracting x and y values
X_right = right_click_data.values[:, :42]  # First 42 columns as features
y_right = np.zeros((X_right.shape[0], 1)) 

X_left = left_click_data.values[:, :42]
y_left = np.ones((X_left.shape[0], 1))  

X_general = general_data.values[:, :42]
y_general = np.full((X_general.shape[0], 1), 2) 

# Combine datasets
X = np.vstack((X_right, X_left, X_general))
y = np.vstack((y_right, y_left, y_general))

# Normalize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print(np.mean(X, axis=0))
print(np.std(X, axis=0))
# Reshape for CNN input
X = X.reshape(X.shape[0], 21, 2, 1)  # Reshaping to (samples, height, width, channels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(21, 2, 1)),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer for 3 classes
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

model.save('clickIdentification.h5')