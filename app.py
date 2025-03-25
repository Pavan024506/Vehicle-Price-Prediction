import streamlit as st
import pandas as pd
import joblib


model = joblib.load("vehicle_price_predictor.pkl")
st.title("🚗 Vehicle Price Predictor")

st.sidebar.header("Enter Vehicle Details")

make = st.sidebar.text_input("Make (e.g., Toyota)")
model_name = st.sidebar.text_input("Model (e.g., Corolla)")

year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, value=2020)
cylinders = st.sidebar.number_input("Cylinders", min_value=2, max_value=16, value=4)
fuel = st.sidebar.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
mileage = st.sidebar.number_input("Mileage (in miles)", min_value=0, value=30000)
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"])
trim = st.sidebar.text_input("Trim (e.g., SE, LX)")
body = st.sidebar.selectbox("Body Style", ["SUV", "Sedan", "Pickup Truck", "Hatchback", "Coupe", "Van"])
doors = st.sidebar.number_input("Doors", min_value=2, max_value=6, value=4)
exterior_color = st.sidebar.text_input("Exterior Color")
interior_color = st.sidebar.text_input("Interior Color")
drivetrain = st.sidebar.selectbox("Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive"])



input_data = pd.DataFrame({
    'make': [make],
    'model': [model_name],
    'year': [year],
    'cylinders': [cylinders],
    'fuel': [fuel],
    'mileage': [mileage],
    'transmission': [transmission],
    'trim': [trim],
    'body': [body],
    'doors': [doors],
    'exterior_color': [exterior_color],
    'interior_color': [interior_color],
    'drivetrain': [drivetrain]
})


# Dummy encoding (match training structure)
input_encoded = pd.get_dummies(input_data)
missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[model.feature_names_in_]

# Predict price
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Vehicle Price: ${prediction:,.2f}")
