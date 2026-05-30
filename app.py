import streamlit as st
import pandas as pd
import joblib
import traceback

st.set_page_config(page_title="Car Price Prediction")

st.title("🚗 Car Price Prediction using Stacking Regression")

# Load model safely
try:
    model = joblib.load("models/stacking_regressor.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model.")
    st.code(traceback.format_exc())
    st.stop()

# Input Fields
vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=30, value=5)

km_driven = st.number_input(
    "KM Driven",
    min_value=0,
    max_value=500000,
    value=50000
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual", "Trustmark Dealer"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

transmission_type = st.selectbox(
    "Transmission Type",
    ["Manual", "Automatic"]
)

mileage = st.number_input(
    "Mileage",
    min_value=0.0,
    max_value=50.0,
    value=20.0
)

engine = st.number_input(
    "Engine (CC)",
    min_value=500,
    max_value=5000,
    value=1200
)

max_power = st.number_input(
    "Max Power",
    min_value=20.0,
    max_value=500.0,
    value=80.0
)

seats = st.number_input(
    "Seats",
    min_value=2,
    max_value=10,
    value=5
)

brand = st.text_input("Brand", "Maruti")
model_name = st.text_input("Model", "Alto")
car_name = st.text_input("Car Name", "Maruti Alto")

# Predict
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "car_name": [car_name],
        "brand": [brand],
        "model": [model_name],
        "vehicle_age": [vehicle_age],
        "km_driven": [km_driven],
        "seller_type": [seller_type],
        "fuel_type": [fuel_type],
        "transmission_type": [transmission_type],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    try:
        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated Selling Price: ₹ {prediction:,.0f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.code(traceback.format_exc())