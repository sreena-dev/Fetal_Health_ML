import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE # Import SMOTE
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Load Data ---
try:
    df = pd.read_csv('fetal_health.csv')
except FileNotFoundError:
    print("Error: fetal_health.csv not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. Separate features (X) and target (y) ---
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# --- 3. Map target labels to 0-indexed and assign class names ---
# Original labels: 1.0 (Normal), 2.0 (Suspect), 3.0 (Pathological)
# Mapped labels:   0   (Normal), 1   (Suspect), 2   (Pathological)
y_mapped = y - 1

# Create a dictionary for class names for easier plotting
class_names = {
    0: 'Normal',
    1: 'Suspect',
    2: 'Pathological'
}

# --- 4. Perform Train-Test Split (necessary before SMOTE) ---
# Even though we'll only plot the resampled training data,
# a split is typically done before SMOTE to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)

# --- 5. Oversampling Minority Class using SMOTE on Training Data ---
print("--- Applying SMOTE to Training Data ---")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n--- Resampled Training Data Class Distribution (after SMOTE) ---")
print(pd.Series(y_train_res).value_counts())

# --- 6. Scale the Resampled Training Features ---
# Fit scaler on resampled training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)

# --- 7. Dimensionality Reduction using PCA to 3 components on Resampled Data ---
# Apply PCA to the scaled, resampled training data
pca = PCA(n_components=3, random_state=42)
X_pca_resampled = pca.fit_transform(X_train_scaled)

# Create DataFrame for Plotly using the resampled data
df_pca_resampled = pd.DataFrame(data=X_pca_resampled, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
df_pca_resampled['Fetal Health'] = y_train_res.map(class_names) # Map numeric labels to names

# --- 8. Define custom color palette for Plotly ---
custom_colors_plotly = {
    'Pathological': 'rgb(255, 0, 0)',     # Bright Red
    'Normal': 'rgb(0, 100, 0)',           # Dark Green
    'Suspect': 'rgb(255, 255, 0)'         # Bright Yellow
}

# --- 9. Create the Interactive 3D Scatter Plot using Plotly Express ---
fig = px.scatter_3d(
    df_pca_resampled, # Use the DataFrame with resampled data
    x='Principal Component 1',
    y='Principal Component 2',
    z='Principal Component 3',
    color='Fetal Health', # Color points based on 'Fetal Health' column
    color_discrete_map=custom_colors_plotly, # Apply custom colors
    title='Interactive 3D PCA Plot of Fetal Health Data (After SMOTE)',
    hover_name='Fetal Health', # This makes the 'Fetal Health' label appear on hover
    # hover_data=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'] # Optional: Add more data to hover
)

# --- 10. Customize Plotly Layout for Dark Theme and Full Screen ---
fig.update_layout(
    title_font_color='white',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        # Set all background and line colors within the 3D scene to black for a "completely black" look
        xaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", tickfont=dict(color='white'), title_font=dict(color='white')),
        zaxis=dict(backgroundcolor="black", gridcolor="black", showbackground=True, zerolinecolor="black", tickfont=dict(color='white'), title_font=dict(color='white')),
        bgcolor="black" # Background color of the 3D scene itself
    ),
    paper_bgcolor='black', # Background color of the entire figure
    plot_bgcolor='black',  # Background color of the plotting area (less relevant for 3D, but good practice)
    font=dict(color='white'), # Default text color for titles, legends, etc.
    legend_title_font_color='white', # Specific legend title font color
    legend=dict(font=dict(color='white')), # Specific legend item font color
    # Set width and height to None for responsive full-page fit
    width=None,
    height=None,
    autosize=True, # Ensure autosizing is enabled
    margin=dict(l=0, r=0, b=0, t=40) # Adjust margins to minimize white space around the plot
)

# --- 11. Save the plot as an HTML file ---
output_filename = 'fetal_health_3d_pca_interactive_after_smote.html'
fig.write_html(output_filename, full_html=True, include_plotlyjs='cdn')
print(f"Interactive plot saved to {output_filename}. Open this file in your web browser.")
