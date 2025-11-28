import streamlit as st
import pandas as pd
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="California House Price Predictor", layout="centered")

# --- LOAD ENGINE ---
# This function loads the 'brain' (model) and the 'translator' (scaler)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('ridge_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

# --- UI HEADER ---
st.title("🏡 California House Price Predictor")
st.write("Using **Ridge Regression** to estimate property values.")

# Check if model loaded correctly
if model is None:
    st.error(" Model files not found!")
    st.info("Please run '2_train_ridge.py' to generate the model files.")
    st.stop()

# --- INPUT FORM ---
st.subheader("Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (x$10k)", value=3.5, help="3.5 = $35,000")
    house_age = st.number_input("House Age (Years)", value=25.0)
    avg_rooms = st.number_input("Avg Rooms", value=5.0)
    avg_bedrooms = st.number_input("Avg Bedrooms", value=1.0)

with col2:
    pop = st.number_input("Population in Block", value=900.0)
    avg_occ = st.number_input("Avg Occupancy", value=3.0)
    lat = st.number_input("Latitude", value=34.00)
    lon = st.number_input("Longitude", value=-118.00)

# --- PREDICTION LOGIC ---
if st.button("Predict Value", type="primary"):
    # 1. Prepare Data
    input_df = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [avg_rooms],
        'AveBedrms': [avg_bedrooms],
        'Population': [pop],
        'AveOccup': [avg_occ],
        'Latitude': [lat],
        'Longitude': [lon]
    })
    
    # 2. Scale Data (Must use the SAME scaler from training)
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict
    prediction = model.predict(input_scaled)
    
    # 4. Format Output (Target is in $100,000s)
    final_price = prediction[0] * 100000
    
    st.success(f"Estimated Price: **${final_price:,.2f}**")