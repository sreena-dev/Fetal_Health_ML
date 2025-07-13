import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from keras import Sequential

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
try:
    df = pd.read_csv('fetal_health.csv')
except FileNotFoundError:
    print("Error: fetal_health.csv not found. Please ensure the file is in the correct directory.")
    exit()

# print("Columns in the DataFrame:", df.columns.tolist())
# Separate features (X) and target (y)
# 'fetal_health' is the target variable
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Convert target labels to be 0-indexed (0, 1, 2 instead of 1, 2, 3)
# This is a good practice for sparse_categorical_crossentropy
# Original labels: 1.0 (Normal), 2.0 (Suspect), 3.0 (Pathological)
# Mapped labels:   0   (Normal), 1   (Suspect), 2   (Pathological)
y = y - 1

# Split data into training and testing sets
# We use a stratify argument to ensure similar class distribution in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the numerical features
# StandardScaler is highly recommended for DNNs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get number of features and classes
num_features = X_train_scaled.shape[1]
num_classes = len(np.unique(y)) # Should be 3: 0, 1, 2

print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of y_train: {y_train.shape}")

# --- 2. Build the TensorFlow Keras Model (DNN) ---

model = Sequential([
    # Input layer: Automatically infers input_shape from the first layer if not specified here
    # For tabular data, a Dense layer is appropriate
    layers.Input(shape=(num_features,)),

    # First hidden layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), # Dropout to prevent overfitting

    # Second hidden layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # Another dropout layer

    # Third hidden layer (optional, can be adjusted)
    layers.Dense(32, activation='relu'),

    # Output layer:
    # 'num_classes' neurons, one for each fetal health category (0, 1, 2)
    # 'softmax' activation for multi-class classification, outputs probabilities for each class
    layers.Dense(num_classes, activation='softmax')
])

# Display the model summary
model.summary()

# --- 3. Compile the Model ---

model.compile(optimizer='adam', # Adam is a popular and effective optimizer
              loss='sparse_categorical_crossentropy', # Use for integer labels (0, 1, 2)
              metrics=['accuracy']) # Track accuracy during training

# --- 4. Train the Model ---

print("\n--- Training the Model ---")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,          # Number of times to iterate over the entire dataset
    batch_size=32,      # Number of samples per gradient update
    validation_split=0.2, # Use 20% of training data for validation during training
    verbose=1           # Show training progress
)

# --- 5. Evaluate the Model ---

print("\n--- Evaluating the Model on Test Data ---")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- Optional: Make Predictions ---
print("\n--- Making Predictions on a few test samples ---")
predictions_proba = model.predict(X_test_scaled[:5]) # Get probabilities for first 5 test samples
predicted_classes = np.argmax(predictions_proba, axis=1) # Get the class with highest probability

print("Predicted probabilities for first 5 samples:\n", predictions_proba)
print("Predicted classes (0=Normal, 1=Suspect, 2=Pathological):\n", predicted_classes)
print("True classes for first 5 samples:\n", y_test.head(5).values)

# accuracy =  0.9178