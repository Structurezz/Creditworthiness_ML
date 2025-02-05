import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and adapt the dataset for creditworthiness."""
    # Add synthetic financial features
    df["AccountBalance"] = np.random.randint(500, 10000, size=len(df))
    df["SpendingPattern"] = np.random.uniform(0.1, 0.9, size=len(df))
    df["CreditUtilization"] = np.random.uniform(0.2, 0.8, size=len(df))
    df["RepaymentHistory"] = np.random.choice(["Good", "Average", "Poor"], size=len(df), p=[0.7, 0.2, 0.1])

    # Generate Creditworthiness Score
    conditions = [
        (df["MonthlyIncome"] >= 5000) & (df["JobSatisfaction"] >= 3) & (df["RepaymentHistory"] == "Good"),
        (df["MonthlyIncome"] >= 3000) & (df["JobSatisfaction"] >= 2) & (df["RepaymentHistory"] != "Poor"),
        (df["MonthlyIncome"] < 3000) | (df["RepaymentHistory"] == "Poor")
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk"]

    df["Creditworthiness"] = np.select(conditions, choices, default="Medium Risk")
    return df
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
    valid_patterns = ["High", "Low", "Medium"]
    if data["SpendingPattern"].iloc[0] not in valid_patterns:
        raise ValueError("Invalid SpendingPattern. Must be one of: High, Low, Medium.")

    return data

def save_data(df: pd.DataFrame, filepath: str):
    """Saves the processed dataset."""
    df.to_csv(filepath, index=False)
