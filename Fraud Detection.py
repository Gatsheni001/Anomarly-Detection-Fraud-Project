import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load the data

np.random.seed(42)
data = {
    'TransactionId': np.arange(1 , 501),
    'Amount': np.concatenate((
        #Random Arragements for normal transactions
        np.random.normal(100, 10, 400),
        #Random Arragements for fraudulent transactions
        np.random.normal(1000, 100, 100)
        
    )),
    'Merchant_Category': np.random.choice(['Entertainment', 'Grocery', 'Apparel', 'Food', 'Fuel'], 500),
    'Transation_Location': np.random.choice(['Online', 'Store'], 500),
    'User_Behavior': np.random.choice('Frequent', 'Occasional', 'Rare', 500)
    
}

