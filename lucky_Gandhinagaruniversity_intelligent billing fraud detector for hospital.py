import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hospital Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# EXPECTED MODEL FEATURES (EDIT IF NEEDED)
# -----------------------------
EXPECTED_COLUMNS = ["age", "billing_amount", "diagnosis_code"]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# -----------------------------
# PREPROCESS FUNCTION (KEY FIX)
# -----------------------------
def preprocess_input_dataframe(df):
    original_cols = list(df.columns)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Mapping variations
    column_mapping = {
        "patient_age": "age",
        "age_years": "age",
        "amt": "billing_amount",
        "amount": "billing_amount",
        "bill_amount": "billing_amount",
        "total_amount": "billing_amount",
        "diag_code": "diagnosis_code",
        "diagnosis": "diagnosis_code"
    }

    df = df.rename(columns=column_mapping)

    # Track missing columns
    missing_cols = []

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            missing_cols.append(col)

    # Keep only required columns
    df = df[EXPECTED_COLUMNS]

    # Convert to numeric safely
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df, original_cols, missing_cols

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💳 Fraud Detection System")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Manual Prediction", "📂 Bulk Scanner"]
)

st.sidebar.markdown("---")
st.sidebar.info("Detect fraudulent hospital billing claims.")

# -----------------------------
# HOME
# -----------------------------
if page == "🏠 Home":
    st.title("🏥 Hospital Billing Fraud Detection")

    st.markdown("""
    ### Features:
    - 🔍 Manual Prediction
    - 📂 Bulk CSV Fraud Detection
    - 📊 Visual Analytics
    - 📥 Download Results

    Upload hospital billing data and detect fraud instantly.
    """)

# -----------------------------
# MANUAL PREDICTION
# -----------------------------
elif page == "🔍 Manual Prediction":

    st.title("🔍 Manual Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 120, 30)

    with col2:
        billing_amount = st.number_input("Billing Amount", 0.0, 1000000.0, 1000.0)

    with col3:
        diagnosis_code = st.selectbox("Diagnosis Code", [0, 1, 2, 3])

    input_data = np.array([[age, billing_amount, diagnosis_code]])

    if st.button("🚀 Predict"):
        try:
            pred = model.predict(input_data)[0]

            if pred == 1:
                st.error("🚨 Fraud Detected")
            else:
                st.success("✅ Legitimate Claim")

        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)

# -----------------------------
# BULK SCANNER
# -----------------------------
elif page == "📂 Bulk Scanner":

    st.title("📂 Bulk Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("📊 Uploaded Data")
            st.dataframe(df.head())

            processed_df, original_cols, missing_cols = preprocess_input_dataframe(df)

            # Show column info
            st.subheader("🧠 Column Handling Info")
            st.write("Uploaded Columns:", original_cols)
            st.write("Expected Columns:", EXPECTED_COLUMNS)

            if missing_cols:
                st.warning(f"⚠️ Missing columns filled with default values: {missing_cols}")

            if st.button("🚀 Run Prediction"):

                predictions = model.predict(processed_df)

                df["Fraud Prediction"] = predictions
                df["Fraud Prediction"] = df["Fraud Prediction"].map({
                    0: "Legitimate",
                    1: "Fraud"
                })

                st.subheader("✅ Results")
                st.dataframe(df)

                # -----------------------------
                # CHARTS
                # -----------------------------
                st.subheader("📊 Fraud Analysis")

                fraud_counts = df["Fraud Prediction"].value_counts()

                # Bar chart
                fig1, ax1 = plt.subplots()
                ax1.bar(fraud_counts.index, fraud_counts.values)
                ax1.set_title("Fraud vs Legitimate Count")
                st.pyplot(fig1)

                # Pie chart
                fig2, ax2 = plt.subplots()
                ax2.pie(fraud_counts.values, labels=fraud_counts.index, autopct='%1.1f%%')
                ax2.set_title("Fraud Distribution")
                st.pyplot(fig2)

                # -----------------------------
                # DOWNLOAD
                # -----------------------------
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Download Results",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error("❌ Error processing file")
            st.exception(e)

    else:
        st.info("Upload a CSV file to begin.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with Streamlit 🚀")
