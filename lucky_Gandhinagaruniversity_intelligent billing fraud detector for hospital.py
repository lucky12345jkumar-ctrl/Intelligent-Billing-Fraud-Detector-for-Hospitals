import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

# Page title
st.title("🏥 Intelligent Hospital Billing Fraud Detector")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    # Encoding categorical column
    df = pd.get_dummies(df, columns=['TreatmentType'], drop_first=True)

    # Feature Engineering
    df['AmountDifference'] = df['BillingAmount'] - df['ApprovedAmount']
    df['BillingRatio'] = df['BillingAmount'] / (df['ApprovedAmount'] + 1)

    # Features and target
    X = df.drop(['FraudFlag', 'ClaimID', 'PatientID', 'ProviderID'], axis=1)
    y = df['FraudFlag']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Train Isolation Forest
    iso = IsolationForest(contamination=0.05)
    df['Anomaly'] = iso.fit_predict(X)

    df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})

    st.success("Model trained successfully!")

    st.subheader("Enter Claim Details for Prediction")

    TreatmentDurationDays = st.number_input("Treatment Duration Days")

    BillingAmount = st.number_input("Billing Amount")

    ApprovedAmount = st.number_input("Approved Amount")

    NumProcedures = st.number_input("Number of Procedures")

    # Calculate engineered features
    AmountDifference = BillingAmount - ApprovedAmount
    BillingRatio = BillingAmount / (ApprovedAmount + 1)

    if st.button("Predict Fraud"):

        input_data = np.array([[

            TreatmentDurationDays,
            BillingAmount,
            ApprovedAmount,
            NumProcedures,
            AmountDifference,
            BillingRatio

        ]])

        # Adjust input columns
        model_columns = X.columns[:len(input_data[0])]

        prediction = rf.predict(input_data)[0]

        anomaly = iso.predict(input_data)[0]
        anomaly = 0 if anomaly == 1 else 1

        final_flag = 1 if (BillingRatio > 2 or anomaly == 1 or prediction == 1) else 0

        if final_flag == 1:
            st.error("🚨 Fraudulent Claim Detected")
        else:
            st.success("✅ Genuine Claim")
