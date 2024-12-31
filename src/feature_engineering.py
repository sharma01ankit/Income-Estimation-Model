import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_affluence_clusters(input_path, output_path):
    """
    Create geolocation-based affluence clusters and assign average income.
    """
    data = pd.read_csv(input_path)
    salaried_data = data[data['IsSalaried'] == 1]

    # Perform KNN clustering for geolocation-based average income
    knn = NearestNeighbors(n_neighbors=5).fit(salaried_data[['Geolocation']])
    distances, indices = knn.kneighbors(data[['Geolocation']])
    
    # Assign cluster-based average income
    cluster_income = []
    for neighbors in indices:
        avg_income = salaried_data.iloc[neighbors]['Income'].mean()
        cluster_income.append(avg_income)
    
    data['ClusterIncome'] = cluster_income
    data['ConfidenceScore'] = 1 / (1 + distances.mean(axis=1))
    
    # Save data with clusters
    data.to_csv(output_path, index=False)
    print(f"Clustered data saved to {output_path}")

def engineer_features(input_path, output_path):
    """
    Create final feature set for modeling.
    """
    data = pd.read_csv(input_path)
    
    # Add additional features
    data['IncomeFlag'] = (data['Income'] > 0).astype(int)  # Known income flag
    data['AgeGroup'] = pd.cut(data['Age'], bins=[18, 30, 40, 50, 60], labels=[0, 1, 2, 3])
    
    # Drop unnecessary columns
    features = data.drop(columns=['CustomerID', 'Geolocation'])
    
    features.to_csv(output_path, index=False)
    print(f"Feature data saved to {output_path}")
