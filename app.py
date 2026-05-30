import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Prediction")

@st.cache_resource
def load_model():
    return joblib.load("models/stacking_regressor.pkl")

model = load_model()

st.title("🚗 Car Price Prediction")
st.write("Stacking Regression using CarDekho Dataset")

vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
km_driven = st.number_input("KM Driven", 0, 500000, 50000)

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

mileage = st.number_input("Mileage", 0.0, 50.0, 20.0)
engine = st.number_input("Engine", 500, 5000, 1200)
max_power = st.number_input("Max Power", 20.0, 500.0, 80.0)
seats = st.number_input("Seats", 2, 10, 5)

brand = st.text_input("Brand", "Maruti")
model_name = st.text_input("Model", "Alto")
car_name = st.text_input("Car Name", "Maruti Alto")

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

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Selling Price: ₹ {prediction:,.0f}"
    )