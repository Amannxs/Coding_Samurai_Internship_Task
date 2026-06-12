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
    area_sqft = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500)
    lot_sqft = st.number_input("Lot Size (sq ft)", min_value=100, max_value=50000, value=2000)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2)
    floors = st.slider("Total Floors", min_value=1, max_value=5, value=1)

with col2:
    st.subheader("📍 Location & Features")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2015)
    garage_spaces = st.slider("Garage Spaces", min_value=0, max_value=5, value=1)
    
    # Checkboxes for Booleans
    has_garden = st.checkbox("Has Garden?")
    has_pool = st.checkbox("Has Pool?")
    
    # Text/Categorical Inputs (Replace options with actual values from your dataset)
    city = st.selectbox("City", ["City A", "City B", "City C"]) 
    neighborhood = st.selectbox("Neighborhood", ["Net A", "Net B"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
st.write("---")

# 4. Prediction Execution Trigger
# 4. Prediction Execution Trigger
if st.button("💰 Calculate Estimated Market Value", type="primary", use_container_width=True):
    
    # A. Map binary/boolean features to 1 and 0
    garden_mapped = 1 if has_garden else 0
    pool_mapped = 1 if has_pool else 0
    
    # B. Map your categories to numeric values (matching your notebook's label encoding)
    # Temporary maps matching typical label encodings (adjust numbers if your notebook used specific values)
    city_map = {"City A": 0, "City B": 1, "City C": 2}
    neighborhood_map = {"Net A": 0, "Net B": 1}
    property_map = {"House": 0, "Apartment": 1, "Condo": 2}
    
    city_encoded = city_map.get(city, 0)
    neighborhood_encoded = neighborhood_map.get(neighborhood, 0)
    property_encoded = property_map.get(property_type, 0)

    # C. Assemble the vector in the EXACT order your notebook processed columns:
    # city, neighborhood, property_type, bedrooms, bathrooms, area_sqft, lot_sqft, year_built, garage_spaces, has_garden, has_pool, floors
    raw_features = np.array([[
        city_encoded, neighborhood_encoded, property_encoded,
        bedrooms, bathrooms, area_sqft, lot_sqft, year_built, 
        garage_spaces, garden_mapped, pool_mapped, floors
    ]])
    
    # D. Transform your features using your saved scaler weights
    final_features = scaler.transform(raw_features)
    
    # E. Run Prediction Engine
    estimated_price = model.predict(final_features)
    
    # F. Render Responsive Output
    st.balloons()
    st.success(f"### Estimated Property Valuation: **${estimated_price[0]:,.2f}**")
