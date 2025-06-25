import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('flight_delay_model.pkl', mmap_mode='r')
label_encoders = joblib.load('label_encoders.pkl')

st.title("Flight Delay Predictor")

carrier = st.selectbox("Unique Carrier", label_encoders['UniqueCarrier'].classes_)
origin = st.selectbox("Origin Airport", label_encoders['Origin'].classes_)
dest = st.selectbox("Destination Airport", label_encoders['Dest'].classes_)
crs_dep_time = st.number_input("Scheduled Departure Time (HHMM)", value=1200)
day_of_week = st.selectbox("Day of the Week (1=Mon, 7=Sun)", list(range(1, 8)))
distance = st.number_input("Distance (miles)", value=500)

# Encode categorical
carrier_encoded = label_encoders['UniqueCarrier'].transform([carrier])[0]
origin_encoded = label_encoders['Origin'].transform([origin])[0]
dest_encoded = label_encoders['Dest'].transform([dest])[0]

features = np.array([[carrier_encoded, crs_dep_time, day_of_week, distance, origin_encoded, dest_encoded]])

if st.button("Predict Delay"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ Flight is likely to be delayed.")
    else:
        st.success("✅ Flight is likely to be on time.")
