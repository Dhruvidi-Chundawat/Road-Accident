import streamlit as st
import joblib

# Load ML components
model = joblib.load("accident_model.pkl")
scaler = joblib.load("scaler.pkl")
le_city = joblib.load("city_encoder.pkl")
le_cause = joblib.load("cause_encoder.pkl")
le_sub = joblib.load("sub_encoder.pkl")
le_outcome = joblib.load("outcome_encoder.pkl")

st.title("🚦 Road Accident Outcome Prediction")

count = st.number_input("Accident Count", min_value=0)

city = st.selectbox("City", sorted(le_city.classes_))
cause = st.selectbox("Cause Category", sorted(le_cause.classes_))
sub = st.selectbox("Cause Subcategory", sorted(le_sub.classes_))

if st.button("Predict"):

    city_enc = le_city.transform([city])[0]
    cause_enc = le_cause.transform([cause])[0]
    sub_enc = le_sub.transform([sub])[0]

    accident_risk = count / 1000
    city_freq = count
    cause_severity = count

    input_data = scaler.transform(
        [[count, cause_enc, sub_enc, city_enc,
          accident_risk, city_freq, cause_severity]]
    )

    pred = model.predict(input_data)
    result = le_outcome.inverse_transform(pred)

    st.success(f"Predicted Outcome: {result[0]}")