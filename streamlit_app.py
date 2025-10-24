import streamlit as st
import pandas as pd
import joblib

# load model and scaler.
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('preprocessor.pkl')
    return model, scaler

kmeans_model, scaler = load_model_and_scaler()

# defining input columns.
input_columns = [
    'balance', 'balance_frequency', 'purchases', 'oneoff_purchases',
    'installments_purchases', 'cash_advance', 'purchases_frequency',
    'oneoff_purchases_frequency', 'purchases_installments_frequency',
    'cash_advance_frequency', 'cash_advance_trx', 'purchases_trx',
    'credit_limit', 'payments', 'minimum_payments', 'prc_full_payment', 'tenure'
]

# split columns, for readability
group1 = input_columns[:6]
group2 = input_columns[6:12]
group3 = input_columns[12:]

st.title("Credit Card Customer Clustering (K-Means)")

with st.form("input_form"):
    st.subheader("Group 1 — Balances & Purchases")
    col1, col2 = st.columns(2)
    with col1:
        balance = st.number_input("Balance", min_value=0.0, step=100.0)
        balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.5)
        purchases = st.number_input("Purchases", min_value=0.0, step=100.0)
    with col2:
        oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, step=100.0)
        installments_purchases = st.number_input("Installments Purchases", min_value=0.0, step=100.0)
        cash_advance = st.number_input("Cash Advance", min_value=0.0, step=100.0)

    st.subheader("Group 2 — Purchase & Cash Frequencies")
    col3, col4 = st.columns(2)
    with col3:
        purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
        oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.5)
        purchases_installments_frequency = st.slider("Purchases Installments Frequency", 0.0, 1.0, 0.5)
    with col4:
        cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.5)
        cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0.0, step=1.0)
        purchases_trx = st.number_input("Purchases Transactions", min_value=0.0, step=1.0)

    st.subheader("Group 3 — Credit & Payments")
    col5, col6 = st.columns(2)
    with col5:
        credit_limit = st.number_input("Credit Limit", min_value=0.0, step=500.0)
        payments = st.number_input("Payments", min_value=0.0, step=100.0)
        minimum_payments = st.number_input("Minimum Payments", min_value=0.0, step=50.0)
    with col6:
        prc_full_payment = st.slider("Percent of Full Payment", 0.0, 1.0, 0.5)
        tenure = st.number_input("Tenure (months)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    # build DataFrame for model input
    input_values = [
        balance, balance_frequency, purchases, oneoff_purchases,
        installments_purchases, cash_advance, purchases_frequency,
        oneoff_purchases_frequency, purchases_installments_frequency,
        cash_advance_frequency, cash_advance_trx, purchases_trx,
        credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]

    df = pd.DataFrame([input_values], columns=input_columns)

    # scale and predict
    scaled_df = scaler.transform(df)
    cluster = kmeans_model.predict(scaled_df)[0]

    if cluster == 0:
        st.success(f"✅ Predicted Cluster: 0, you are an 'Active consumer'")
    else:
        st.success(f"✅ Predicted Cluster: 1, you are a 'Cautious buyer'")