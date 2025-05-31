# 💳 Advanced Credit Card Fraud Detection System

This is a powerful Streamlit web app for detecting fraudulent credit card transactions using machine learning (XGBoost), explainable AI (SHAP), and interactive visualizations (Plotly).

## 🔍 Features

- Real-time fraud prediction based on custom transaction input
- Interactive SHAP plots for model explainability
- Map visualization of transaction location
- Batch fraud prediction on CSV file upload
- Fraud probability analysis with beautiful UI/UX
- Built using XGBoost, SHAP, Plotly, and Streamlit

## 🚀 Dataset
The dataset used for this project is the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset?resource=download&select=credit_card_transactions.csv)
dataset from Kaggle.

## 🚀 Deployment

This app is deployed using [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own:

1. **Fork or clone this repo**
2. Add your trained model file at `model/xgb_model.json`
3. Make sure `requirements.txt` includes all dependencies
4. Deploy it via Streamlit Cloud using `app.py` as the entry point

## 🧠 Model

The model used is a trained `XGBoost` classifier saved as `xgb_model.json`. It predicts fraud likelihood based on transaction amount, time, user profile, and category.

## 📁 Project Structure

```
.
├── app.py
├── model/
│   └── xgb_model.json
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

## 📦 Installation (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```
