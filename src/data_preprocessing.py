# src/data_preprocessing.py
import pandas as pd

def validate_and_preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Validate and preprocess the input data for the model.
    
    Args:
        input_data (dict): A dictionary of input features.

    Returns:
        pd.DataFrame: A processed dataframe ready for model prediction.
    """
    # Expected features
    required_features = ["MonthlyIncome", "AccountBalance", "SpendingPattern", "CreditUtilization"]

    # Check for missing features
    for feature in required_features:
        if feature not in input_data:
            raise ValueError(f"Missing feature: {feature}")

    # Convert to dataframe
    data = pd.DataFrame([input_data])

    # Validate feature types and ranges
    data["MonthlyIncome"] = pd.to_numeric(data["MonthlyIncome"], errors="coerce").fillna(0)
    data["AccountBalance"] = pd.to_numeric(data["AccountBalance"], errors="coerce").fillna(0)
    data["CreditUtilization"] = pd.to_numeric(data["CreditUtilization"], errors="coerce").clip(0, 1)

    # Validate SpendingPattern
    valid_patterns = {"High": 2, "Medium": 1, "Low": 0}
    if data["SpendingPattern"].iloc[0] not in valid_patterns:
        raise ValueError("Invalid SpendingPattern. Must be one of: High, Low, Medium.")
    
    # Map SpendingPattern to numeric
    data["SpendingPattern"] = valid_patterns[data["SpendingPattern"].iloc[0]]

    return data
def validate_and_preprocess_input(data):
    """Validate and preprocess the input data."""
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")

    required_fields = ["MonthlyIncome", "AccountBalance", "SpendingPattern", "CreditUtilization"]

    # Ensure all required fields are present
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Convert to numeric types
    try:
        processed_data = {
            "MonthlyIncome": float(data["MonthlyIncome"]),
            "AccountBalance": float(data["AccountBalance"]),
            "SpendingPattern": float(data["SpendingPattern"]),
            "CreditUtilization": float(data["CreditUtilization"]),
        }
    except ValueError:
        raise ValueError("All fields must be numeric.")

    return processed_data

