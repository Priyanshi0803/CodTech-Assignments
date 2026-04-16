import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Loan Predictor", layout="wide")

st.title("💰 Loan Approval Prediction App")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("User Input")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", [0, 1])

# -------------------------------
# Prediction (Single Input)
# -------------------------------
if st.button("Predict"):
    features = np.array([[income, loan_amount, credit_history]])

    try:
        result = model.predict(features)

        if result[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("⚠️ Make sure model is trained with SAME 3 features")

# -------------------------------
# 📊 Data Visualizations
# -------------------------------
st.subheader("📊 Data Insights")

try:
    df = pd.read_csv("train.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Income Distribution")
        st.bar_chart(df['ApplicantIncome'])

    with col2:
        st.write("### Loan Amount Trend")
        st.line_chart(df['LoanAmount'])

except:
    st.warning("⚠️ Dataset not found. Check file path.")

# -------------------------------
# 🔥 Feature Importance
# -------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("🔥 Feature Importance")
    st.bar_chart(model.feature_importances_)

# -------------------------------
# 📂 CSV Upload for Bulk Prediction
# -------------------------------
st.subheader("📂 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # ✅ Select only required columns
        required_cols = ["ApplicantIncome", "LoanAmount", "Credit_History"]

        if not all(col in data.columns for col in required_cols):
            st.error("❌ CSV must contain: ApplicantIncome, LoanAmount, Credit_History")
        else:
            data = data[required_cols]

            # Handle missing values
            data.fillna(0, inplace=True)

            # Convert to numeric
            data = data.apply(pd.to_numeric, errors='coerce')

            # Fill NaN after conversion
            data.fillna(0, inplace=True)

            preds = model.predict(data)

            # Show results
            data["Prediction"] = preds
            data["Prediction"] = data["Prediction"].map({1: "Approved", 0: "Rejected"})

            st.write("### Predictions")
            st.dataframe(data)

            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")