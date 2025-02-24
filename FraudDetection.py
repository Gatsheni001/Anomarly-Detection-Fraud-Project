import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Custom CSS for Enhanced UI
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #fff;
            font-family: Arial, sans-serif;
        }
        .stButton > button {
            background-color: #ff007f;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 10px;
        }
        .stButton > button:hover {
            background-color: #ff00ff;
        }
        .stDataFrame {
            border: 2px solid #ff007f;
            border-radius: 5px;
            padding: 5px;
        }
        .css-1d391kg {
            background-color: #222222 !important;
            border-radius: 10px;
            padding: 15px;
        }
        .metric-container {
            background: linear-gradient(135deg, #ff007f, #00ff99);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI Setup
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Real-Time Fraud Monitoring Dashboard")
add_custom_css()

# File Upload Section
st.sidebar.header("Upload Your Transactions CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    # Ensure required columns exist
    expected_columns = {'Transaction_ID', 'Amount', 'Merchant_Category', 'Transaction_Location', 'User_Behavior'}
    if not expected_columns.issubset(df.columns):
        st.sidebar.error("Uploaded CSV must contain: Transaction_ID, Amount, Merchant_Category, Transaction_Location, User_Behavior")
        st.stop()

    # Step 2: Apply Isolation Forest for Anomaly Detection
    iso_forest = IsolationForest(n_estimators=100, contamination=0.04, random_state=42)
    df['Anomaly_Score'] = iso_forest.fit_predict(df[['Amount']])
    df['Fraudulent'] = df['Anomaly_Score'].apply(lambda x: 'Fraud' if x == -1 else 'Legit')

    detected_fraud = df[df['Fraudulent'] == 'Fraud']

    # Sidebar Filters
    st.sidebar.header("Filter Transactions")
    merchant_filter = st.sidebar.multiselect("Select Merchant Category:", df['Merchant_Category'].unique(), default=df['Merchant_Category'].unique())
    location_filter = st.sidebar.multiselect("Select Transaction Location:", df['Transaction_Location'].unique(), default=df['Transaction_Location'].unique())
    behavior_filter = st.sidebar.multiselect("Select User Behavior:", df['User_Behavior'].unique(), default=df['User_Behavior'].unique())
    amount_range = st.sidebar.slider("Select Amount Range:", int(df['Amount'].min()), int(df['Amount'].max()), (int(df['Amount'].min()), int(df['Amount'].max())))

    # Apply Filters
    filtered_df = df[(df['Merchant_Category'].isin(merchant_filter)) &
                     (df['Transaction_Location'].isin(location_filter)) &
                     (df['User_Behavior'].isin(behavior_filter)) &
                     (df['Amount'].between(amount_range[0], amount_range[1]))]

    # Fraud Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">Total Transactions: {}</div>'.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">Fraudulent Transactions: {}</div>'.format(len(detected_fraud)), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">Fraud Percentage: {:.2f}%</div>'.format((len(detected_fraud)/len(df)*100)), unsafe_allow_html=True)

    # Display Data
    st.write("## Transactions Overview")
    st.dataframe(filtered_df)

    st.write("## Detected Fraudulent Transactions")
    st.dataframe(detected_fraud)

    # Visualization with Plotly
    st.write("## Fraud Detection Visualization")
    fig = px.scatter(df, x='Transaction_ID', y='Amount', color='Fraudulent',
                     title="Transaction Amounts with Fraud Highlighted",
                     color_discrete_map={'Legit': 'blue', 'Fraud': 'red'})
    st.plotly_chart(fig)
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()
