# app.py
import streamlit as st
import pickle
import numpy as np

# Load model and encoders
with open('car_price_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
le_fuel = data['le_fuel']
le_seller = data['le_seller']
le_trans = data['le_trans']

st.title("🚗 Car Price Prediction App")
st.markdown("Predict your car's selling price using Gradient Boosting Regressor.")

# User Inputs
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2018)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("KMs Driven", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", le_fuel.classes_)
seller_type = st.selectbox("Seller Type", le_seller.classes_)
transmission = st.selectbox("Transmission", le_trans.classes_)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

if st.button("Predict Price"):
    try:
        fuel_encoded = le_fuel.transform([fuel_type])[0]
        seller_encoded = le_seller.transform([seller_type])[0]
        trans_encoded = le_trans.transform([transmission])[0]

        input_data = np.array([[year, present_price, kms_driven, fuel_encoded,
                                seller_encoded, trans_encoded, owner]])

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} lakhs")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Gradient Boosting Regressor")
