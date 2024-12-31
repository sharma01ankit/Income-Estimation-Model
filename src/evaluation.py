import pandas as pd
import json

def evaluate_models(model_path, test_data_path, output_path):
    """
    Evaluate model performance and save metrics.
    """
    # Dummy metrics for example
    metrics = {
        "Existing-to-bank customers": 0.7,
        "New-to-bank customers": 0.6,
        "New-to-bank customers with no bureau history": 0.55,
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Evaluation metrics saved to {output_path}")
