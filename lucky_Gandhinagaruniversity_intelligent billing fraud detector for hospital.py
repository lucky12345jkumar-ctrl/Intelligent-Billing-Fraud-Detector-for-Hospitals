import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="FraudGuard AI", layout="wide")

st.title("🛡️ FraudGuard AI - Hospital Billing Fraud Detection")

# -------------------------------
# Load models (NO retraining)
# -------------------------------
rf = joblib.load("rf_model.pkl")
iso = joblib.load("iso_model.pkl")
expected_features = joblib.load("features.pkl")

# -------------------------------
# Preprocessing (LOCKED PIPELINE)
# -------------------------------
def preprocess_input(df):
    df = df.copy()

    # Feature engineering
    if 'BillingAmount' in df.columns and 'ApprovedAmount' in df.columns:
        df['AmountDifference'] = df['BillingAmount'] - df['ApprovedAmount']
        df['BillingRatio'] = df['BillingAmount'] / df['ApprovedAmount'].replace(0, 1)

    # Encoding
    if 'TreatmentType' in df.columns:
        df = pd.get_dummies(df, columns=['TreatmentType'], drop_first=True)

    # Align schema
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    return df

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["🔍 Manual Prediction", "📂 Bulk Scanner"])

# ======================================
# 🔍 MANUAL PREDICTION
# ======================================
with tab1:

    st.subheader("Enter Claim Details")

    BillingAmount = st.number_input("Billing Amount", min_value=1.0)
    ApprovedAmount = st.number_input("Approved Amount", min_value=1.0)
    NumProcedures = st.number_input("Number of Procedures", min_value=1)
    TreatmentDurationDays = st.number_input("Treatment Duration (Days)", min_value=1)

    if st.button("Predict Fraud"):

        input_df = pd.DataFrame([{
            "BillingAmount": BillingAmount,
            "ApprovedAmount": ApprovedAmount,
            "NumProcedures": NumProcedures,
            "TreatmentDurationDays": TreatmentDurationDays
        }])

        input_df = preprocess_input(input_df)

        pred = rf.predict(input_df)[0]
        prob = rf.predict_proba(input_df)[0][1]

        anomaly = iso.predict(input_df)[0]
        anomaly = 0 if anomaly == 1 else 1

        final_score = (prob + anomaly) / 2

        st.subheader("Result")

        if final_score > 0.5:
            st.error(f"🚨 Fraud Detected (Risk Score: {final_score:.2f})")
        else:
            st.success(f"✅ Genuine Claim (Risk Score: {1-final_score:.2f})")

# ======================================
# 📂 BULK SCANNER
# ======================================
with tab2:

    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview", df.head())

        required_cols = ["BillingAmount", "ApprovedAmount"]

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:

            if st.button("Run Scan"):

                processed = preprocess_input(df)

                preds = rf.predict(processed)
                probs = rf.predict_proba(processed)[:, 1]

                anomaly = iso.predict(processed)
                anomaly = np.where(anomaly == 1, 0, 1)

                final_score = (probs + anomaly) / 2

                df["Fraud_Prediction"] = preds
                df["Fraud_Probability"] = probs
                df["Risk_Score"] = final_score

                st.subheader("Results")
                st.dataframe(df)

                # Metrics
                st.metric("Fraud Rate", f"{df['Fraud_Prediction'].mean()*100:.2f}%")

                # Download
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "📥 Download Results",
                    csv,
                    "fraud_results.csv",
                    "text/csv"
                )
