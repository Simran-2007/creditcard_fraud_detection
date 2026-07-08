"""
Credit Card Fraud Detection — Enhanced Interactive App
--------------------------------------------------------
Same model as before, now with:
  - Custom gradient theme + animated cards
  - Live risk gauge meter (plotly)
  - Real-time fraud alert popup
  - CSV + PDF report download
  - Sidebar navigation

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import io

st.set_page_config(page_title="Fraud Shield", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #f5f3ff 0%, #eef2ff 45%, #f0f9ff 100%);
    background-attachment: fixed;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b69 0%, #764ba2 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 8px;
}

.block-container {
    padding-top: 2rem;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 10px 30px rgba(118,75,162,0.25);
}
.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.3rem 0 0 0;
}
.metric-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 20px rgba(118,75,162,0.12);
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
}
.metric-card .label {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 500;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 0.2rem;
}
.fraud-alert {
    background: linear-gradient(135deg, #ff5f6d 0%, #ffc371 100%);
    color: white;
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    font-size: 1.1rem;
    font-weight: 600;
    animation: pulseGlow 1.2s ease-in-out 3;
    margin-top: 1rem;
}
.safe-alert {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
    color: white;
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 1rem;
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(255,95,109,0.6); }
    70% { box-shadow: 0 0 0 18px rgba(255,95,109,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,95,109,0); }
}
div.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: opacity 0.2s ease;
}
div.stButton > button:hover {
    opacity: 0.9;
    color: white;
}
</style>
""", unsafe_allow_html=True)


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

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ Fraud Shield</h1>
    <p>Real-time credit card fraud detection, powered by your trained model</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🔍 Check Transaction", "📊 Dashboard", "📄 Reports"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Session stats**")
total_checked = len(st.session_state.history)
fraud_flagged = sum(1 for h in st.session_state.history if h["prediction"] == "FRAUD")
st.sidebar.metric("Checked", total_checked)
st.sidebar.metric("Fraud flagged", fraud_flagged)

# ---------------------------------------------------------------------------
# PAGE 1 — Check a transaction
# ---------------------------------------------------------------------------
if page == "🔍 Check Transaction":
    st.subheader("Enter transaction details")
    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        inputs["customer_age"] = st.number_input("Customer age", 18, 100, 35)
        inputs["customer_gender"] = st.selectbox("Gender", list(encoders["customer_gender"].classes_))
        inputs["city"] = st.selectbox("City", list(encoders["city"].classes_))
        inputs["card_type"] = st.selectbox("Card type", list(encoders["card_type"].classes_))
        inputs["transaction_amount"] = st.number_input("Transaction amount (₹)", 0.0, 1_000_000.0, 5000.0)

    with col2:
        inputs["merchant_category"] = st.selectbox("Merchant category", list(encoders["merchant_category"].classes_))
        inputs["transaction_hour"] = st.slider("Transaction hour (0-23)", 0, 23, 12)
        inputs["transaction_day"] = st.selectbox("Day of week", list(encoders["transaction_day"].classes_))
        inputs["is_weekend"] = st.checkbox("Is weekend?")
        inputs["num_transactions_today"] = st.number_input("Transactions today", 0, 100, 3)

    with col3:
        inputs["distance_from_home_km"] = st.number_input("Distance from home (km)", 0.0, 20000.0, 15.0)
        inputs["is_international"] = st.checkbox("International transaction?")
        inputs["pin_used"] = st.checkbox("PIN used?", value=True)
        inputs["account_balance"] = st.number_input("Account balance (₹)", 0.0, 50_000_000.0, 150000.0)
        inputs["previous_fraud_flag"] = st.checkbox("Previous fraud flag on account?")

    check = st.button("🔍 Check transaction", type="primary", use_container_width=True)

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

        gauge_col, alert_col = st.columns([1, 1.4])

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(fraud_prob * 100, 1),
                title={"text": "Fraud risk score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#764ba2"},
                    "steps": [
                        {"range": [0, 40], "color": "#a8e063"},
                        {"range": [40, 70], "color": "#ffc371"},
                        {"range": [70, 100], "color": "#ff5f6d"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with alert_col:
            if prediction == 1:
                st.toast("🚨 Fraud detected on this transaction!", icon="🚨")
                st.markdown(
                    f'<div class="fraud-alert">🚨 FRAUD DETECTED — {fraud_prob*100:.2f}% fraud probability.'
                    f'<br>Transaction blocked.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.toast("✅ Transaction looks normal", icon="✅")
                st.markdown(
                    f'<div class="safe-alert">✅ NORMAL TRANSACTION — {probability[0]*100:.2f}% confidence.'
                    f'<br>Transaction approved.</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# PAGE 2 — Dashboard
# ---------------------------------------------------------------------------
elif page == "📊 Dashboard":
    st.subheader("Session dashboard")

    if not st.session_state.history:
        st.info("No transactions checked yet. Go to 'Check Transaction' and run one.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        total = len(hist_df)
        fraud_count = int((hist_df["prediction"] == "FRAUD").sum())
        avg_risk = hist_df["fraud_probability_%"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in zip(
            [c1, c2, c3, c4],
            ["Checked", "Fraud flagged", "Fraud rate", "Avg. risk"],
            [total, fraud_count, f"{(fraud_count/total*100):.1f}%", f"{avg_risk:.1f}%"],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("####")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=hist_df["fraud_probability_%"],
            mode="lines+markers",
            line=dict(color="#764ba2", width=3),
            fill="tozeroy",
            fillcolor="rgba(118,75,162,0.15)",
        ))
        fig.update_layout(
            title="Fraud probability over checks",
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            height=280,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Transaction history**")
        st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE 3 — Reports (CSV + PDF download)
# ---------------------------------------------------------------------------
else:
    st.subheader("Download reports")

    if not st.session_state.history:
        st.info("No transactions checked yet. There's nothing to report.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        total = len(hist_df)
        fraud_count = int((hist_df["prediction"] == "FRAUD").sum())

        st.markdown(f"This session has **{total}** transactions checked, **{fraud_count}** flagged as fraud.")

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV report",
            data=csv_bytes,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        def build_pdf(df, total, fraud_count):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 12, "Credit Card Fraud Detection Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 8, f"Transactions checked: {total}", ln=True)
            pdf.cell(0, 8, f"Fraud flagged: {fraud_count}", ln=True)
            pdf.cell(0, 8, f"Fraud rate: {(fraud_count/total*100):.1f}%", ln=True)
            pdf.ln(6)

            pdf.set_font("Helvetica", "B", 9)
            headers = ["Time", "Amount", "City", "Merchant", "Result", "Risk %"]
            widths = [22, 25, 28, 35, 22, 20]
            for h, w in zip(headers, widths):
                pdf.cell(w, 8, h, border=1)
            pdf.ln()

            pdf.set_font("Helvetica", "", 8)
            for _, row in df.iterrows():
                values = [
                    str(row["time"]),
                    f"{row['amount']:.0f}",
                    str(row["city"])[:14],
                    str(row["merchant"])[:16],
                    str(row["prediction"]),
                    f"{row['fraud_probability_%']:.1f}",
                ]
                for v, w in zip(values, widths):
                    pdf.cell(w, 7, v, border=1)
                pdf.ln()

            return bytes(pdf.output())

        pdf_bytes = build_pdf(hist_df, total, fraud_count)
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
