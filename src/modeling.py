import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def train_models(input_path, output_path):
    """
    Train hybrid ensemble models (e.g., Enhanced KNN, ANN).
    """
    data = pd.read_csv(input_path)
    
    # Separate features and target
    X = data.drop(columns=['Income'])
    y = data['Income']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Random Forest (proxy for Enhanced KNN)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # Model 2: Artificial Neural Network (ANN)
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
    ann.fit(X_train, y_train)
    ann_preds = ann.predict(X_test)
    
    # Evaluate models
    rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
    ann_rmse = mean_squared_error(y_test, ann_preds, squared=False)
    
    print(f"RF RMSE: {rf_rmse}, ANN RMSE: {ann_rmse}")
    
    # Save models (if required)
    # pickle.dump(rf, open(f"{output_path}/rf_model.pkl", "wb"))
    # pickle.dump(ann, open(f"{output_path}/ann_model.pkl", "wb"))
