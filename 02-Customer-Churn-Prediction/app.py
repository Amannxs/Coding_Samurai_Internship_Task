import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 💡 CRUCIAL STREAMLIT RULE: Must be the first execution command!
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ==========================================
# 1. CACHE & LOAD PIPELINE MODEL
# ==========================================
@st.cache_resource
def load_churn_pipeline():
    # Path to your downloaded .pkl file from Kaggle output
    model_path = 'final_customer_churn_model.pkl'
    
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        return pipeline
    else:
        st.error(f"⚠️ Model file '{model_path}' not found! Please place it in the same directory.")
        return None

pipeline = load_churn_pipeline()

# ==========================================
# 2. STREAMLIT USER INTERFACE (UI)
# ==========================================
st.title("📉 Telecom Customer Churn Predictor")
st.write("Enter the customer metrics below to analyze their cancellation probability score.")

st.write("---")

# Creating double column layouts for a clean dashboard look
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Months)", min_value=1, max_value=120, value=24)
    usage_freq = st.number_input("Usage Frequency (Times/Month)", min_value=0, max_value=100, value=15)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
    payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=30, value=1)

with col2:
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
    last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, max_value=30, value=5)
    
    # Text Categoricals mapping seamlessly to numeric values as expected by the pipeline
    gender = st.selectbox("Gender", options=["Male", "Female"])
    contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
    subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])

# Map input strings back to structural numbers exactly matching cell [77] mappings
contract_mapping = {'Monthly': 1, 'Quarterly': 6, 'Annual': 12}
sub_mapping = {'Premium': 1, 'Standard': 2, 'Basic': 3}
gender_mapping = {'Male': 0, 'Female': 1}

# ==========================================
# 3. PREDICTION PIPELINE EXECUTION
# ==========================================
if st.button("Predict Churn Risk", type="primary"):
    if pipeline is not None:
        # Construct DataFrame matching the baseline columns array format
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender_mapping[gender],
            'Tenure': tenure,
            'Usage Frequency': usage_freq,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay,
            'Subscription Type': sub_mapping[subscription_type],
            'Contract Length': contract_mapping[contract_length],
            'Total Spend': total_spend,
            'Last Interaction': last_interaction
        }])
        
        # Predict Class and Probability
        prediction = pipeline.predict(input_data)[0]
        
        # Note: LightGBM pipeline supports predict_proba if initialized properly 
        try:
            prob = pipeline.predict_proba(input_data)[0][1] * 100
        except AttributeError:
            prob = None

        st.write("---")
        # Display explicit evaluation results block
        if prediction == 1:
            st.error("### 🚨 Prediction: HIGH CHURN RISK (Customer Likely to Leave)")
            if prob is not None:
                st.metric(label="Churn Risk Probability Score", value=f"{prob:.2f}%")
        else:
            st.success("### ✅ Prediction: LOYAL CUSTOMER (Customer Likely to Stay)")
            if prob is not None:
                st.metric(label="Retention Confidence Score", value=f"{(100 - prob):.2f}%")