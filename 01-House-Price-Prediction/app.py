import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration for responsiveness and wide layouts
st.set_page_config(
    page_title="House Valuation Engine",
    page_icon="🏡",
    layout="wide"
)

# 1. Load the exported model and scaler safely
# 1. Load the exported model and scaler safely with correct folder paths
@st.cache_resource
def load_ml_components():
    # Added folder prefix so Streamlit Cloud can find the files
    with open('01-House-Price-Prediction/lr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('01-House-Price-Prediction/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_ml_components()

# 2. App Title & Branding
st.title("🏡 Core Property Valuation Engine")
st.markdown("Enter the foundational structural elements of the home to generate an instant market price estimate.")
st.write("---")

# 3. Responsive Two-Column Input Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📐 Dimensions & Layout")
    area = st.number_input("Total Area (sq ft)", min_value=300, max_value=10000, value=1800, step=50)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)
    floors = st.slider("Total Floors", min_value=1, max_value=4, value=1)

with col2:
    st.subheader("📍 Location & Quality")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2018, step=1)
    condition = st.slider("Overall Condition Rating", min_value=1, max_value=5, value=3)
    
    # Categorical fields matching your dataset's exact string categories
    location = st.selectbox("Location Setting", ["Urban", "Suburb", "Rural"])
    garage = st.selectbox("Does it have a Garage?", ["Yes", "No"])

st.write("---")

# 4. Prediction Execution Trigger
if st.button("💰 Calculate Estimated Market Value", type="primary", use_container_width=True):
    
    # A. Preprocess Categorical Fields to match your Notebook
    # Map Garage string to binary numeric
    garage_mapped = 1 if garage == "Yes" else 0
    
    # Manually match your exact One-Hot encoding array shape
    # Adjust flags depending on whether your pipeline generated 2 or 3 binary columns for Location
    is_suburb = 1 if location == "Suburb" else 0
    is_urban = 1 if location == "Urban" else 0
    
    # B. Assemble the Numerical Vector for the Scaler
    # Must be in the precise original order your notebook fed into StandardScaler
    raw_numerical_features = np.array([[area, bedrooms, bathrooms, floors, year_built, condition, garage_mapped]])
    
    # C. Transform using the saved scaler weights
    scaled_numerical = scaler.transform(raw_numerical_features)
    
    # D. Reconstruct the full 11-column matching matrix (Scaled numericals + encoded locations)
    # This aligns perfectly with your training data's structural layout
    final_features = np.append(scaled_numerical, [[is_suburb, is_urban]], axis=1)
    
    # E. Run Prediction Engine
    estimated_price = model.predict(final_features)
    
    # F. Render Responsive Output
    st.balloons()
    st.success(f"### Estimated Property Valuation: **${estimated_price[0]:,.2f}**")
