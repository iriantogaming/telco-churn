
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
st.markdown("Prediksi apakah pelanggan akan berhenti berlangganan (Churn) atau tidak.")

with st.sidebar:
    st.header("📊 Informasi Model")
    st.write("Algoritma: Random Forest")
    st.write("Accuracy: 80.13%")
    st.write("Dataset: Telco Customer Churn")
    st.write("Jumlah Data: 7043")
    st.info("💵 Semua nilai biaya menggunakan mata uang USD ($)")

st.subheader("Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
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

    phone_service = st.selectbox(
        "Phone Service",
        encoders["Phone Service"].classes_
    )

    internet_service = st.selectbox(
        "Internet Service",
        encoders["Internet Service"].classes_
    )

with col2:
    contract = st.selectbox(
        "Contract",
        encoders["Contract"].classes_
    )

    paperless = st.selectbox(
        "Paperless Billing",
        encoders["Paperless Billing"].classes_
    )

    payment_method = st.selectbox(
        "Payment Method",
        encoders["Payment Method"].classes_
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=360,
        value=12
    )

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        format="%.2f"
    )

    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=1000.0,
        format="%.2f"
    )

if st.button("🔍 Predict Churn", use_container_width=True):

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

    if "Phone Service" in input_data:
        input_data["Phone Service"] = encoders["Phone Service"].transform([phone_service])[0]

    if "Internet Service" in input_data:
        input_data["Internet Service"] = encoders["Internet Service"].transform([internet_service])[0]

    if "Contract" in input_data:
        input_data["Contract"] = encoders["Contract"].transform([contract])[0]

    if "Paperless Billing" in input_data:
        input_data["Paperless Billing"] = encoders["Paperless Billing"].transform([paperless])[0]

    if "Payment Method" in input_data:
        input_data["Payment Method"] = encoders["Payment Method"].transform([payment_method])[0]

    if "Tenure Months" in input_data:
        input_data["Tenure Months"] = tenure

    if "Monthly Charges" in input_data:
        input_data["Monthly Charges"] = monthly

    if "Total Charges" in input_data:
        input_data["Total Charges"] = total

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    st.subheader("📋 Ringkasan Input")

    st.write(f"Monthly Charges : ${monthly:,.2f}")
    st.write(f"Total Charges : ${total:,.2f}")

    if prediction == 1:
        st.error("⚠️ Customer Diprediksi Akan Churn")
        st.write("Pelanggan berpotensi berhenti menggunakan layanan.")
    else:
        st.success("✅ Customer Diprediksi Tidak Churn")
        st.write("Pelanggan diperkirakan tetap menggunakan layanan.")

    st.subheader("Data yang Dimasukkan")
    st.dataframe(input_df, use_container_width=True)




