import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration for professional responsiveness
st.set_page_config(
    page_title="House Valuation Engine",
    page_icon="🏡",
    layout="wide"
)

# 1. Load the exported model and scaler safely (Configured for Streamlit Cloud paths)
@st.cache_resource
def load_ml_components():
    with open('01-House-Price-Prediction/lr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('01-House-Price-Prediction/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_ml_components()
except FileNotFoundError:
    st.error("⚠️ Model components missing. Ensure 'lr_model.pkl' and 'scaler.pkl' are inside the '01-House-Price-Prediction' folder.")

# 2. App Title & Interface Header
st.title("🏡 Core Property Valuation Engine")
st.markdown("Enter the foundational structural and locational elements of the property to generate an instant market price estimate.")
st.write("---")

# 3. Responsive Two-Column Input Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📐 Dimensions & Layout")
    area_sqft = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1500, step=50)
    lot_sqft = st.number_input("Lot Size (sq ft)", min_value=100, max_value=100000, value=2000, step=50)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2)
    floors = st.slider("Total Floors", min_value=1, max_value=5, value=1)

with col2:
    st.subheader("📍 Location & Features")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2015, step=1)
    garage_spaces = st.slider("Garage Spaces", min_value=0, max_value=5, value=1)
    
    # Toggle switches / Checkboxes for Booleans
    has_garden = st.checkbox("Has Garden?")
    has_pool = st.checkbox("Has Pool?")
    
    # Categorical Dropdowns matching dataset classifications
    city = st.selectbox("City", ["Austin", "Dallas", "Houston", "San Antonio"])
    neighborhood = st.selectbox("Neighborhood", ["Downtown", "Suburbs", "Rural Area"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

st.write("---")

# 4. Prediction Execution Logic
if st.button("💰 Calculate Estimated Market Value", type="primary", use_container_width=True):
    
    # A. Map binary/boolean parameters to 1 or 0
    garden_mapped = 1 if has_garden else 0
    pool_mapped = 1 if has_pool else 0
    
    # B. Explicitly Label-Encode Categorical fields to match the Kaggle index logic
    city_map = {"Austin": 0, "Dallas": 1, "Houston": 2, "San Antonio": 3}
    neighborhood_map = {"Downtown": 0, "Suburbs": 1, "Rural Area": 2}
    property_map = {"House": 0, "Apartment": 1, "Condo": 2}
    
    city_encoded = city_map.get(city, 0)
    neighborhood_encoded = neighborhood_map.get(neighborhood, 0)
    property_encoded = property_map.get(property_type, 0)

    # C. Extract ONLY the 7 features your StandardScaler expects:
    # Structure: [bedrooms, bathrooms, area_sqft, lot_sqft, year_built, garage_spaces, floors]
    numerical_features = np.array([[
        bedrooms, bathrooms, area_sqft, lot_sqft, year_built, garage_spaces, floors
    ]])
    
    # D. Transform the 7 structural values using your notebook's scale configurations
    scaled_numerical = scaler.transform(numerical_features)
    
    # E. Construct the final 12-feature matrix layout in the EXACT order your notebook trained it:
    # ['city', 'neighborhood', 'property_type', 'bedrooms', 'bathrooms', 'area_sqft', 'lot_sqft', 'year_built', 'garage_spaces', 'has_garden', 'has_pool', 'floors']
    final_features = np.array([[
        city_encoded,
        neighborhood_encoded,
        property_encoded,
        scaled_numerical[0][0],  # scaled bedrooms
        scaled_numerical[0][1],  # scaled bathrooms
        scaled_numerical[0][2],  # scaled area_sqft
        scaled_numerical[0][3],  # scaled lot_sqft
        scaled_numerical[0][4],  # scaled year_built
        scaled_numerical[0][5],  # scaled garage_spaces
        garden_mapped,
        pool_mapped,
        scaled_numerical[0][6]   # scaled floors
    ]])
    
    # F. Run the linear regression model prediction
    estimated_price = model.predict(final_features)
    
    # G. Render Responsive User Deliverables
    st.balloons()
    st.success(f"### Estimated Property Valuation: **${estimated_price[0]:,.2f}**")
