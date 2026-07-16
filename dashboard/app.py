"""
Customer Churn Risk Dashboard
------------------------------
Run locally with:  streamlit run app.py

Expects these files in the same folder (produced by
00_save_dashboard_artifacts.py):
    churn_model.pkl
    scaler.pkl
    model_columns.pkl
    scaler_columns.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

# ---------------------------------------------------------------
# Load saved artifacts (cached so they only load once per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    # Pull the exact columns each object was actually fit on, straight
    # from the fitted object itself — this can never go out of sync
    # with the model, unlike a separately saved column list.
    model_columns = list(model.feature_names_in_)
    scaler_columns = list(scaler.feature_names_in_)
    explainer = shap.TreeExplainer(model)
    return model, scaler, model_columns, scaler_columns, explainer

model, scaler, model_columns, scaler_columns, explainer = load_artifacts()

st.title("📉 Customer Churn Risk Dashboard")
st.caption("Predict churn risk for a customer and see exactly what's driving it.")

# ---------------------------------------------------------------
# Sidebar: customer inputs
# ---------------------------------------------------------------
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5
)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges)
)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
)

paperless_billing = st.sidebar.checkbox("Paperless Billing", value=True)
senior_citizen = st.sidebar.checkbox("Senior Citizen")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
phone_service = st.sidebar.checkbox("Phone Service", value=True)
multiple_lines = st.sidebar.checkbox("Multiple Lines", value=False) if phone_service else False

st.sidebar.subheader("Household")
partner = st.sidebar.checkbox("Has Partner")
dependents = st.sidebar.checkbox("Has Dependents")

# Add-ons only make sense if the customer actually has internet
addon_disabled = internet_service == "No"
st.sidebar.subheader("Add-On Services" + (" (requires internet)" if addon_disabled else ""))
online_security = st.sidebar.checkbox("Online Security", disabled=addon_disabled)
online_backup = st.sidebar.checkbox("Online Backup", disabled=addon_disabled)
device_protection = st.sidebar.checkbox("Device Protection", disabled=addon_disabled)
tech_support = st.sidebar.checkbox("Tech Support", disabled=addon_disabled)
streaming_tv = st.sidebar.checkbox("Streaming TV", disabled=addon_disabled)
streaming_movies = st.sidebar.checkbox("Streaming Movies", disabled=addon_disabled)

total_addons = sum([
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
])

# ---------------------------------------------------------------
# Build a household type label, matching how it was engineered originally
# ---------------------------------------------------------------
if partner and dependents:
    household_type = "Partner + Dependents"
elif partner and not dependents:
    household_type = "Partner only"
elif not partner and dependents:
    household_type = "Dependents only"
else:
    household_type = "Neither"

# ---------------------------------------------------------------
# Assemble the raw input row, matching original (pre-encoding) columns
# ---------------------------------------------------------------
raw_input = {
    "SeniorCitizen": 1 if senior_citizen else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "TotalAddOns": total_addons,
    "gender_Male": 1 if gender == "Male" else 0,
    "PhoneService_Yes": 1 if phone_service else 0,
    "MultipleLines_Yes": 1 if multiple_lines else 0,
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,
    "OnlineSecurity_Yes": 1 if online_security else 0,
    "OnlineBackup_Yes": 1 if online_backup else 0,
    "DeviceProtection_Yes": 1 if device_protection else 0,
    "TechSupport_Yes": 1 if tech_support else 0,
    "StreamingTV_Yes": 1 if streaming_tv else 0,
    "StreamingMovies_Yes": 1 if streaming_movies else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaperlessBilling_Yes": 1 if paperless_billing else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "HouseholdType_Neither": 1 if household_type == "Neither" else 0,
    "HouseholdType_Partner + Dependents": 1 if household_type == "Partner + Dependents" else 0,
    "HouseholdType_Partner only": 1 if household_type == "Partner only" else 0,
}

input_df = pd.DataFrame([raw_input])

# ---------------------------------------------------------------
# Scale the numeric columns exactly as done during training
# ---------------------------------------------------------------
input_df[scaler_columns] = scaler.transform(input_df[scaler_columns])

# ---------------------------------------------------------------
# Align to the model's exact training columns (drops any extras,
# adds any missing as 0) so the feature set matches perfectly
# ---------------------------------------------------------------
input_final = input_df.reindex(columns=model_columns, fill_value=0)

# ---------------------------------------------------------------
# Predict
# ---------------------------------------------------------------
churn_probability = model.predict_proba(input_final)[0, 1]
threshold = 0.25  # matches the tuned threshold selected during modeling
prediction_label = "High Risk" if churn_probability >= threshold else "Low Risk"

# ---------------------------------------------------------------
# Display results
# ---------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Churn Risk Score")
    st.metric("Predicted Churn Probability", f"{churn_probability:.1%}")
    if prediction_label == "High Risk":
        st.error(f"⚠️ {prediction_label} — recommend proactive retention outreach")
    else:
        st.success(f"✅ {prediction_label}")
    st.caption(f"Classification threshold: {threshold:.0%} (tuned to prioritize catching at-risk customers)")

with col2:
    st.subheader("What's Driving This Score?")
    shap_values = explainer.shap_values(input_final)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif len(np.array(shap_values).shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    contributions = pd.DataFrame({
        "Feature": model_columns,
        "Impact": sv,
    }).sort_values("Impact", key=abs, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in contributions["Impact"]]
    ax.barh(contributions["Feature"], contributions["Impact"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on churn risk (red = increases risk, blue = decreases risk)")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption(
    "This tool is for illustrative/portfolio purposes, built on the IBM Telco Customer Churn dataset. "
    "Predictions reflect patterns in historical data and should support, not replace, human judgment in retention decisions."
)
