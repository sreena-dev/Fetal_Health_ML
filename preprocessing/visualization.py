import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go # Not strictly needed for px, but good practice

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

# --- 4. Scale features (entire dataset) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Dimensionality Reduction using PCA to 3 components ---
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for Plotly
df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
df_pca['Fetal Health'] = y_mapped.map(class_names) # Map numeric labels to names

# --- 6. Define custom color palette for Plotly ---
custom_colors_plotly = {
    'Pathological': 'red',
    'Normal': 'darkgreen',
    'Suspect': 'yellow'
}

# --- 7. Create the Interactive 3D Scatter Plot using Plotly Express ---
fig = px.scatter_3d(
    df_pca,
    x='Principal Component 1',
    y='Principal Component 2',
    z='Principal Component 3',
    color='Fetal Health', # Color points based on 'Fetal Health' column
    color_discrete_map=custom_colors_plotly, # Apply custom colors
    title='Interactive 3D PCA Plot of Fetal Health Data',
    hover_name='Fetal Health', # This makes the 'Fetal Health' label appear on hover
    # You can add more columns to hover_data if you want to see other PCA values
    # hover_data=['Principal Component 1', 'Principal Component 2', 'Principal Component 3']
)

# --- 8. Customize Plotly Layout for Dark Theme ---
fig.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="white", tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="white", tickfont=dict(color='white'), title_font=dict(color='white')),
        zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="white", tickfont=dict(color='white'), title_font=dict(color='white')),
        # Set the background of the 3D scene itself
        bgcolor="black"
    ),
    paper_bgcolor='black', # Background color of the entire figure
    plot_bgcolor='black',  # Background color of the plotting area
    font=dict(color='white'), # Default text color for titles, legends, etc.
    title_font_color='white', # Specific title font color
    legend_title_font_color='white', # Specific legend title font color
    legend=dict(font=dict(color='white')) # Specific legend item font color
)

# --- 9. Save the plot as an HTML file ---
output_filename = 'fetal_health_3d_pca_interactive.html'
fig.write_html(output_filename)
print(f"Interactive plot saved to {output_filename}. Open this file in your web browser.")

# Removed fig.show() as it might be causing the "site can't be reached" issue
# fig.show()
