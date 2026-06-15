
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📱",
    layout="centered"
)

model = joblib.load("telco_model.pkl")
encoders = joblib.load("telco_encoders.pkl")

st.title("📱 Telco Customer Churn Prediction")
st.write("Prediksi apakah pelanggan akan churn atau tidak.")

gender = st.selectbox(
    "Gender",
    encoders["Gender"].classes_
)

senior = st.selectbox(
    "Senior Citizen",
    encoders["Senior Citizen"].classes_
)

partner = st.selectbox(
    "Partner",
    encoders["Partner"].classes_
)

dependents = st.selectbox(
    "Dependents",
    encoders["Dependents"].classes_
)

tenure = st.slider(
    "Tenure Months",
    0,
    100,
    12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

if st.button("Predict Churn"):

    input_data = {}

    for col in model.feature_names_in_:
        input_data[col] = 0

    if "Gender" in input_data:
        input_data["Gender"] = encoders["Gender"].transform([gender])[0]

    if "Senior Citizen" in input_data:
        input_data["Senior Citizen"] = encoders["Senior Citizen"].transform([senior])[0]

    if "Partner" in input_data:
        input_data["Partner"] = encoders["Partner"].transform([partner])[0]

    if "Dependents" in input_data:
        input_data["Dependents"] = encoders["Dependents"].transform([dependents])[0]

    if "Tenure Months" in input_data:
        input_data["Tenure Months"] = tenure

    if "Monthly Charges" in input_data:
        input_data["Monthly Charges"] = monthly

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer Diprediksi Akan Churn")
    else:
        st.success("✅ Customer Diprediksi Tidak Churn")
