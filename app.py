import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Loan Dashboard", page_icon="💰", layout="wide")

# -------------------------------
# Load Model & Image (FIXED)
# -------------------------------
model = pickle.load(open("Model/ML_Model1.pkl", "rb"))
image = Image.open("bank.png")

# -------------------------------
# Header
# -------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image(image, width=100)
with col2:
    st.title("🏦 Loan Approval Dashboard")
    st.caption("AI-powered credit decision system")

st.markdown("---")

# -------------------------------
# Applicant Details
# -------------------------------
st.subheader("👤 Applicant Details")

a1, a2, a3 = st.columns(3)

with a1:
    name = st.text_input("Full Name")
    age = st.number_input("Age", 18, 70, 30)

with a2:
    phone = st.text_input("Phone")
    email = st.text_input("Email")

with a3:
    applicant_id = st.text_input("ID Number")
    address = st.text_area("Address")

# -------------------------------
# Co-Applicant Details
# -------------------------------
st.subheader("👥 Co-Applicant Details")

has_coapplicant = st.checkbox("Add Co-Applicant")

if has_coapplicant:
    c1, c2, c3 = st.columns(3)

    with c1:
        co_name = st.text_input("Co-Applicant Name")
        co_age = st.number_input("Co-Applicant Age", 18, 70, 28)

    with c2:
        co_phone = st.text_input("Co-Applicant Phone")
        co_job = st.text_input("Occupation")

    with c3:
        relationship = st.selectbox("Relationship", ["Spouse", "Parent", "Sibling", "Other"])
        co_email = st.text_input("Co-Applicant Email")

st.markdown("---")

# -------------------------------
# Loan Inputs
# -------------------------------
st.subheader("📋 Loan & Financial Information")

c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

with c2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income (₹)", value=50000)

with c3:
    coapplicant_income = st.number_input("Coapplicant Income (₹)", value=15000)
    loan_amount = st.number_input("Loan Amount (₹)", value=150000)
    loan_term = st.number_input("Loan Term (months)", value=360)

interest_rate = st.number_input("Interest Rate (%)", value=8.5)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Preprocess
# -------------------------------
def preprocess():
    g = 1 if gender == "Male" else 0
    m = 1 if married == "Yes" else 0
    d = 3 if dependents == "3+" else int(dependents)
    e = 1 if education == "Graduate" else 0
    emp = 1 if employed == "Yes" else 0
    pa = 2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0

    ai = applicant_income * 10
    cai = coapplicant_income * 10
    la = loan_amount / 10

    return np.array([[g, m, d, e, emp, ai, cai, la,
                      loan_term, credit_history, pa]])

# -------------------------------
# EMI
# -------------------------------
def calculate_emi(P, r, n):
    r = r / (12 * 100)
    return (P * r * (1 + r)**n) / ((1 + r)**n - 1)

# -------------------------------
# Max Eligible Loan
# -------------------------------
def get_max_eligible_loan(model, input_data):
    base = input_data.copy()
    current = int(base[0][7])

    if model.predict(base)[0] == 1:
        max_loan = current
        for val in range(current, current * 3, 500):
            temp = base.copy()
            temp[0][7] = val
            if model.predict(temp)[0] == 1:
                max_loan = val
            else:
                break
    else:
        max_loan = 0
        for val in range(current, 0, -500):
            temp = base.copy()
            temp[0][7] = val
            if model.predict(temp)[0] == 1:
                max_loan = val
                break

    return max_loan

# -------------------------------
# Feature Impact
# -------------------------------
def get_feature_impact(model, input_data):
    features = ["Gender","Married","Dependents","Education","Employed",
                "ApplicantIncome","CoapplicantIncome","LoanAmount",
                "Loan_Amount_Term","Credit_History","Property_Area"]

    base = model.predict(input_data)[0]
    impacts = []

    for i in range(len(features)):
        temp = input_data.copy()
        temp[0][i] = 0 if i < 5 else np.mean(input_data)
        impacts.append(base - model.predict(temp)[0])

    return pd.DataFrame({"Feature": features, "Impact": impacts})

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Analyze Application"):

    if name.strip() == "":
        st.warning("Enter applicant name")
        st.stop()

    if has_coapplicant and co_name.strip() == "":
        st.warning("Enter co-applicant name")
        st.stop()

    input_data = preprocess()

    if credit_history == 0.0:
        prediction = 0
        prob = 0.05
    else:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.75

    st.markdown("---")
    st.subheader(f"📄 Application Summary - {name}")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if credit_history == 0.0:
        st.warning("⚠️ Rejected due to poor credit history")

    st.write(f"**Loan Requested:** ₹{loan_amount:,}")

    if has_coapplicant:
        st.markdown("### 👥 Co-Applicant")
        st.write(f"{co_name} ({relationship})")

    st.subheader("📊 Approval Probability")
    st.progress(int(prob * 100))
    st.write(f"{prob*100:.2f}% chance")

    st.subheader("💰 EMI")
    emi = calculate_emi(loan_amount, interest_rate, loan_term)
    st.info(f"₹{int(emi):,} per month")

    st.subheader("💡 Eligible Loan")
    eligible = get_max_eligible_loan(model, input_data)

    if eligible == 0:
        st.error("No eligible loan")
    else:
        st.success(f"Up to ₹{int(eligible * 10):,}")

    st.subheader("📊 Insights")

    impact_df = get_feature_impact(model, input_data)

    g1, g2 = st.columns(2)

    with g1:
        fig1, ax1 = plt.subplots()
        colors = ["green" if i > 0 else "red" for i in impact_df["Impact"]]
        ax1.barh(impact_df["Feature"], impact_df["Impact"], color=colors)
        ax1.set_title("Feature Impact")
        st.pyplot(fig1)

    with g2:
        total_income = applicant_income + coapplicant_income
        loan_burden = loan_amount / 12
        remaining = max(total_income - loan_burden, 0)

        fig2, ax2 = plt.subplots()
        ax2.pie(
            [loan_burden, remaining],
            labels=["Loan Burden", "Remaining Income"],
            autopct='%1.1f%%'
        )
        ax2.set_title("Financial Load")
        st.pyplot(fig2)

    if prediction == 0:
        st.subheader("❗ Top Reasons")

        if credit_history == 0.0:
            st.error("🚫 Credit history is poor (major factor)")

        reasons = impact_df[impact_df["Impact"] < 0].sort_values(by="Impact").head(3)
        for r in reasons["Feature"]:
            st.warning(f"❌ {r}")

# Footer
st.markdown("---")
st.caption("🏦 Smart Loan Dashboard | Streamlit")
