import pandas as pd

def preprocess_data(input_path, output_path):
    """
    Load and clean the raw data.
    """
    data = pd.read_csv(input_path)
    
    # Fill missing income for unsalaried customers with a placeholder (to be estimated later)
    data['Income'] = data['Income'].fillna(0)
    
    # Encode categorical data (e.g., Marital Status)
    data['MaritalStatus'] = data['MaritalStatus'].map({'Single': 0, 'Married': 1})
    
    # Save preprocessed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
