import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import time
import base64

# Load model
model = xgb.XGBClassifier()
model.load_model("model/xgb_model.json")

# App Configuration
st.set_page_config(page_title="🚨 Credit Card Fraud Detection Pro", layout="wide", page_icon="💳")
st.title("💳 Advanced AI-Powered Credit Card Fraud Detection System")

# Sidebar - User Input
st.sidebar.header("🧾 Input Transaction Details")

amount = st.sidebar.slider("Transaction Amount ($)", 1, 1000000, 100)
hour = st.sidebar.slider("Hour of Transaction (0–23)", 0, 23, 12)
age = st.sidebar.slider("Customer Age", 18, 100, 35)
category = st.sidebar.selectbox("Merchant Category", ['gas_transport','grocery_pos','shopping_net','travel','misc_pos','health_fitness','entertainment'])
gender = st.sidebar.radio("Gender", ["M", "F"])
city_pop = st.sidebar.slider("Customer City Population", 100, 10000000, 50000)
job = st.sidebar.selectbox("Customer Job", ['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Nurse', 'Artist', 'Freelancer'])
transaction_date = st.sidebar.date_input("Transaction Date", datetime.date.today())

# Encoders
cat_map = {v: i for i, v in enumerate(['gas_transport','grocery_pos','shopping_net','travel','misc_pos','health_fitness','entertainment'])}
gender_map = {'M':0, 'F':1}
job_map = {v: i for i, v in enumerate(['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Nurse', 'Artist', 'Freelancer'])}

weekday = transaction_date.weekday()
is_weekend = 1 if weekday >= 5 else 0
location_distance = round(np.random.uniform(0.1, 5.0), 2)

# Feature Engineering
data = {
    "amount_log": np.log(amount + 1),
    "category": cat_map[category],
    "gender": gender_map[gender],
    "city_pop_log": np.log(city_pop + 1),
    "job": job_map[job],
    "age": age,
    "hour": hour,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "location_distance": location_distance
}

input_df = pd.DataFrame([data])
pred_prob = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

# Prediction Display
col1, col2 = st.columns(2)
with col1:
    st.metric("🧠 Prediction", "Fraud" if pred else "Not Fraud", delta=f"{pred_prob*100:.2f} %")
    if pred_prob > 0.8:
        st.error("⚠️ High Fraud Risk")
    elif pred_prob > 0.5:
        st.warning("⚠️ Moderate Fraud Risk")
    else:
        st.success("✅ Low Fraud Risk")

with col2:
    fig = px.pie(values=[pred_prob, 1-pred_prob], names=["Fraud", "Legit"],
                 color_discrete_sequence=["red", "green"],
                 title="Fraud Probability")
    st.plotly_chart(fig, use_container_width=True)

# Transaction Map (Mock Data)
st.subheader("📍 Transaction Location (Mock Coordinates)")
map_df = pd.DataFrame({
    'lat': [36.01, 36.07],
    'lon': [-82.04, -81.17],
    'name': ['Merchant', 'Customer']
})
fig_map = px.scatter_mapbox(map_df, lat='lat', lon='lon', color='name',
                            mapbox_style="carto-positron", zoom=5)
st.plotly_chart(fig_map)

# SHAP Value Explanation
st.subheader("📊 SHAP Explanation (Why this prediction?)")
explainer = shap.Explainer(model)
shap_values = explainer(input_df)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=7, show=False)
st.pyplot(fig)

# Batch Prediction
st.subheader("📁 Batch Fraud Detection from CSV")
file = st.file_uploader("Upload CSV file with required features", type=["csv"])
if file:
    df = pd.read_csv(file)
    if all(col in df.columns for col in input_df.columns):
        df['prediction'] = model.predict(df)
        df['probability'] = model.predict_proba(df)[:, 1]

        fraud_count = int(df['prediction'].sum())
        legit_count = len(df) - fraud_count

        st.success(f"🔍 Total Fraudulent Transactions: {fraud_count}")
        st.info(f"✅ Total Legitimate Transactions: {legit_count}")

        fig_bar = px.histogram(df, x='probability', nbins=50, title="Prediction Confidence Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(df.head())
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button("⬇️ Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")
    else:
        st.error("Uploaded CSV missing required columns.")

# Footer
st.markdown("""
---
Made with ❤️ by Cybersecurity & AI Experts | Powered by XGBoost + SHAP\n\n GitHub: [d3vpr0x](https://github.com/d3vpr0x/Credit-Card-Fraud-Detection)
""")
