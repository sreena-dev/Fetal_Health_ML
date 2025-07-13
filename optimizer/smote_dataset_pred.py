import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing (from previous steps) ---

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
y_mapped = y - 1

# Create a dictionary for class names
class_names = {
    0: 'Normal',
    1: 'Suspect',
    2: 'Pathological'
}
# Create a list of class names in order for classification report
target_names_list = [class_names[i] for i in sorted(class_names.keys())]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)

# --- 2. Oversampling Minority Class using SMOTE on Training Data ---
print("--- Applying SMOTE to Training Data ---")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n--- Resampled Training Data Class Distribution (after SMOTE) ---")
print(pd.Series(y_train_res).value_counts())


# --- 3. Scale the Resampled Training Features and Original Test Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res) # Fit scaler on resampled training data
X_test_scaled = scaler.transform(X_test)         # Transform original test data

# --- 4. Prepare Data for TensorFlow ---
# Convert NumPy arrays to TensorFlow Datasets for efficient processing
BATCH_SIZE = 32 # Define batch size for training

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_res)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 5. Build the MLP Model with TensorFlow Keras ---
# The input shape is the number of features after scaling
input_shape = X_train_scaled.shape[1]
num_classes = len(class_names)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(input_shape,)), # Input layer
    keras.layers.Dense(128, activation='relu'), # First hidden layer with ReLU activation
    keras.layers.Dropout(0.3), # Dropout for regularization
    keras.layers.Dense(64, activation='relu'),  # Second hidden layer
    keras.layers.Dropout(0.3), # Another dropout layer
    keras.layers.Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class classification
])

# --- 6. Compile the Model ---
# Use Adam optimizer and SparseCategoricalCrossentropy loss
model.compile(
    optimizer='adam', # Adam optimizer as requested
    loss='sparse_categorical_crossentropy', # Suitable for integer labels (0, 1, 2)
    metrics=['accuracy'] # Track accuracy during training
)

model.summary() # Print a summary of the model architecture

# --- 7. Train the Model ---
print("\n--- Training the MLP Model ---")
history = model.fit(
    train_dataset,
    epochs=50, # Number of training epochs
    validation_data=test_dataset, # Evaluate on test data after each epoch
    verbose=1 # Show training progress
)

# --- 8. Evaluate the Model ---
print("\n--- Evaluating the Model on Test Data ---")
loss, accuracy = model.evaluate(test_dataset, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 9. Generate Detailed Classification Metrics ---
print("\n--- Generating Classification Report and Confusion Matrix ---")

# Get predictions from the model
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names_list))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting the Confusion Matrix for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_list, yticklabels=target_names_list)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- 10. Plot Training History (Loss and Accuracy) ---
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# 89% accuracy