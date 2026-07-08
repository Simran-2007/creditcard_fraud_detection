"""
Credit Card Fraud Detection — Interactive App
------------------------------------------------
Loads the model YOU trained in the notebook (fraud_model.pkl, scaler.pkl,
encoders.pkl) and lets you check new transactions through a real form,
with a running dashboard of everything checked this session.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    categorical_cols = joblib.load("categorical_cols.pkl")
    return model, scaler, encoders, feature_columns, categorical_cols


try:
    model, scaler, encoders, feature_columns, categorical_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "Missing model files. Run the notebook's 'STEP 9C: SAVE MODEL' cell, "
        "download the 5 .pkl files, and place them in the same folder as app.py."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("💳 Credit Card Fraud Detection")
st.caption("Live predictions using your trained model — not a static notebook cell.")

tab1, tab2 = st.tabs(["🔍 Check a Transaction", "📊 Dashboard"])

# ---------------------------------------------------------------------------
# TAB 1 — Check a transaction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter Transaction Details")
    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        inputs["customer_age"] = st.number_input("Customer Age", 18, 100, 35)
        inputs["customer_gender"] = st.selectbox("Gender", list(encoders["customer_gender"].classes_))
        inputs["city"] = st.selectbox("City", list(encoders["city"].classes_))
        inputs["card_type"] = st.selectbox("Card Type", list(encoders["card_type"].classes_))
        inputs["transaction_amount"] = st.number_input("Transaction Amount (₹)", 0.0, 1_000_000.0, 5000.0)

    with col2:
        inputs["merchant_category"] = st.selectbox("Merchant Category", list(encoders["merchant_category"].classes_))
        inputs["transaction_hour"] = st.slider("Transaction Hour (0-23)", 0, 23, 12)
        inputs["transaction_day"] = st.selectbox("Day of Week", list(encoders["transaction_day"].classes_))
        inputs["is_weekend"] = st.checkbox("Is Weekend?")
        inputs["num_transactions_today"] = st.number_input("Transactions Today (this account)", 0, 100, 3)

    with col3:
        inputs["distance_from_home_km"] = st.number_input("Distance from Home (km)", 0.0, 20000.0, 15.0)
        inputs["is_international"] = st.checkbox("International Transaction?")
        inputs["pin_used"] = st.checkbox("PIN Used?", value=True)
        inputs["account_balance"] = st.number_input("Account Balance (₹)", 0.0, 50_000_000.0, 150000.0)
        inputs["previous_fraud_flag"] = st.checkbox("Previous Fraud Flag on Account?")

    check = st.button("🔍 Check Transaction", type="primary", use_container_width=True)

    if check:
        row = []
        for col in feature_columns:
            val = inputs[col]
            if col in categorical_cols:
                val = encoders[col].transform([val])[0]
            elif isinstance(val, bool):
                val = int(val)
            row.append(val)

        transaction_df = pd.DataFrame([row], columns=feature_columns)
        transaction_scaled = scaler.transform(transaction_df)

        prediction = model.predict(transaction_scaled)[0]
        probability = model.predict_proba(transaction_scaled)[0]
        fraud_prob = probability[1]

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "amount": inputs["transaction_amount"],
            "city": inputs["city"],
            "merchant": inputs["merchant_category"],
            "card_type": inputs["card_type"],
            "prediction": "FRAUD" if prediction == 1 else "NORMAL",
            "fraud_probability_%": round(fraud_prob * 100, 2),
        })

        st.divider()
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED** — {fraud_prob*100:.2f}% fraud probability. Transaction BLOCKED.")
        else:
            st.success(f"✅ **NORMAL TRANSACTION** — {probability[0]*100:.2f}% confidence. Transaction APPROVED.")

# ---------------------------------------------------------------------------
# TAB 2 — Dashboard / history
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Session Dashboard")

    if not st.session_state.history:
        st.info("No transactions checked yet. Go to 'Check a Transaction' and run one.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        total = len(hist_df)
        fraud_count = int((hist_df["prediction"] == "FRAUD").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Checked", total)
        c2.metric("Fraud Flagged", fraud_count)
        c3.metric("Fraud Rate (this session)", f"{(fraud_count/total*100):.1f}%")

        st.line_chart(hist_df["fraud_probability_%"], height=250)

        st.markdown("**Transaction History**")
        st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()