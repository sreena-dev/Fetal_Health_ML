import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif # Import for feature selection
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

print(f"Original number of features: {X_train.shape[1]}")

# --- Feature Selection (Integrated Step) ---
# When to do it: After initial train/test split, but BEFORE scaling.
# Why: To select relevant features based on training data only, preventing data leakage.

# Define the number of top features to select. You can experiment with this value.
k_features_to_select = 15 # Example: selecting the top 15 features

# Initialize SelectKBest with f_classif for classification tasks
selector = SelectKBest(score_func=f_classif, k=k_features_to_select)

# Fit the selector ONLY on the training data and transform both train and test sets
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the names of the selected features for inspection
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices].tolist()

print(f"\n--- Feature Selection Results ---")
print(f"Selected {len(selected_feature_names)} features using SelectKBest (f_classif):")
print(selected_feature_names)
print(f"Shape of X_train after feature selection: {X_train_selected.shape}")
print(f"Shape of X_test after feature selection: {X_test_selected.shape}")

# --- Scaling the Selected Features ---
# StandardScaler is highly recommended for DNNs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected) # Fit and transform selected training features
X_test_scaled = scaler.transform(X_test_selected)     # Transform selected test features

# Get number of features and classes after selection
num_features = X_train_scaled.shape[1] # This will now be k_features_to_select
num_classes = len(np.unique(y)) # Should be 3: 0, 1, 2

print(f"\nNumber of features after selection and scaling: {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of y_train: {y_train.shape}")

# --- 2. Build the TensorFlow Keras Model (DNN) ---

model = Sequential([
    # Input layer: The input shape must match the number of selected features
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

print(f"\nTest Loss (with feature selection): {loss:.4f}")
print(f"Test Accuracy (with feature selection): {accuracy:.4f}")

# --- Optional: Make Predictions ---
print("\n--- Making Predictions on a few test samples ---")
predictions_proba = model.predict(X_test_scaled[:5]) # Get probabilities for first 5 test samples
predicted_classes = np.argmax(predictions_proba, axis=1) # Get the class with highest probability

print("Predicted probabilities for first 5 samples:\n", predictions_proba)
print("Predicted classes (0=Normal, 1=Suspect, 2=Pathological):\n", predicted_classes)
print("True classes for first 5 samples:\n", y_test.head(5).values)