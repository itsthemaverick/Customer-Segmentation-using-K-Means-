import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data: pd.DataFrame):
    data = data.drop(columns=["CustomerID"])
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features
