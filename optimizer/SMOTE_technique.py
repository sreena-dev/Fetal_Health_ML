import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from keras import Sequential
from imblearn.over_sampling import SMOTE # Import SMOTE
from sklearn.metrics import classification_report, confusion_matrix # For better evaluation

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
try:
    df = pd.read_csv('fetal_health.csv')
except FileNotFoundError:
    print("Error: fetal_health.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Convert target labels to be 0-indexed (0, 1, 2 instead of 1, 2, 3)
y = y - 1

# Split data into training and testing sets
# We use a stratify argument to ensure similar class distribution in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("--- Original Training Data Class Distribution ---")
print(pd.Series(y_train).value_counts())

# --- Oversampling Minority Class using SMOTE ---
# When to do it: After train-test split, but BEFORE scaling.
# Why: SMOTE should only be applied to the training data to prevent data leakage.
# Scaling should happen after SMOTE because SMOTE creates new data points that need scaling.

print("\n--- Applying SMOTE to Training Data ---")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n--- Resampled Training Data Class Distribution (after SMOTE) ---")
print(pd.Series(y_train_res).value_counts())

# --- Scale the Resampled Training Features and Original Test Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res) # Fit scaler on resampled training data
X_test_scaled = scaler.transform(X_test)         # Transform original test data

# Get number of features and classes
num_features = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train_res)) # Should be 3: 0, 1, 2

print(f"\nNumber of features: {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of y_train_res: {y_train_res.shape}")

# --- 2. Build the TensorFlow Keras Model (DNN) ---

model = Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Display the model summary
model.summary()

# --- 3. Compile the Model ---

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. Train the Model ---

print("\n--- Training the Model with SMOTE Resampled Data ---")
history = model.fit(
    X_train_scaled, y_train_res, # Train on resampled data
    epochs=50,
    batch_size=32,
    validation_split=0.2, # Validation split from resampled data
    verbose=1
)

# --- 5. Evaluate the Model ---

print("\n--- Evaluating the Model on Original Test Data ---")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nTest Loss (with SMOTE): {loss:.4f}")
print(f"Test Accuracy (with SMOTE): {accuracy:.4f}")

# --- Additional Evaluation: Classification Report and Confusion Matrix ---
print("\n--- Detailed Classification Report ---")
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

# Define target names for better readability in the report
target_names = ['Normal (0)', 'Suspect (1)', 'Pathological (2)']
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)