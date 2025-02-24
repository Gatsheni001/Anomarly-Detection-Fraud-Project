import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Streamlit UI Setup
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Real-Time Fraud Monitoring Dashboard")

# File Upload Section
st.sidebar.header("Upload Your Transactions CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()

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
st.metric("Total Transactions", len(df))
st.metric("Fraudulent Transactions", len(detected_fraud))
st.metric("Fraud Percentage", f"{(len(detected_fraud)/len(df)*100):.2f}%")

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
