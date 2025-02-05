from src.data_preparation import load_data, preprocess_data, save_data
from src.model_training import train_model
from src.prediction_service import load_model, predict, explain_prediction
from src.utils import ensure_directory
import pandas as pd

# Paths
RAW_DATA_PATH = "data/raw/employee_attrition.csv"
CREDIT_CARD_CUSTOMERS_PATH = "data/raw/credit-card-customers.csv"
CREDITCARD_FRAUD_PATH = "data/raw/creditcard.csv"
CREDIT_SCORE_CLASSIFICATION_PATH = "data/raw/credit_score_classification.csv"
GERMAN_CREDIT_PATH = "data/raw/german_credit.csv"
CREDIT_RISK_DATASET_PATH = "data/raw/credit-risk-dataset.csv"
COMBINED_DATA_PATH = "data/processed/combined_creditworthiness_data.csv"
MODEL_PATH = "models/credit_model.pkl"

def main():
    # Load and process datasets
    raw_data = preprocess_data(load_data(RAW_DATA_PATH))
    processed_credit_card_customers = preprocess_data(load_data(CREDIT_CARD_CUSTOMERS_PATH))
    processed_creditcard_fraud = preprocess_data(load_data(CREDITCARD_FRAUD_PATH))
    processed_credit_score_classification = preprocess_data(load_data(CREDIT_SCORE_CLASSIFICATION_PATH))
    processed_german_credit = preprocess_data(load_data(GERMAN_CREDIT_PATH))
    processed_credit_risk_dataset = preprocess_data(load_data(CREDIT_RISK_DATASET_PATH))

    # Combine datasets
    combined_data = pd.concat([
        raw_data,
        processed_credit_card_customers,
        processed_creditcard_fraud,
        processed_credit_score_classification,
        processed_german_credit,
        processed_credit_risk_dataset
    ], axis=0, ignore_index=True)

    # Debug: Check class distribution in combined data
    print("Class Distribution in Combined Data:")
    if "RiskLevel" in combined_data.columns:
        print(combined_data["RiskLevel"].value_counts())

    # Save combined dataset
    ensure_directory("data/processed/")
    save_data(combined_data, COMBINED_DATA_PATH)

    # Train the model using the combined dataset
    ensure_directory("models/")
    train_model(COMBINED_DATA_PATH, MODEL_PATH)

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Sample input for prediction
    sample_input = {
        "MonthlyIncome": 1000000,
        "AccountBalance": 300990,
        "SpendingPattern": 0.1,
        "CreditUtilization": 0.05
    }

    # Perform prediction
    risk_level, ai_insight = predict(model, sample_input)

    # Display the results
    print(f"Prediction: {risk_level}")
    print(f"AI Insight:\n{ai_insight}")

    # Print raw probabilities for debugging
    raw_probabilities = model.predict_proba([sample_input]) if hasattr(model, "predict_proba") else None
    if raw_probabilities is not None:
        print(f"Raw Probabilities: {raw_probabilities}")

    # Debug: Test with extreme cases
    low_risk_input = {
        "MonthlyIncome": 15000,
        "AccountBalance": 50000,
        "SpendingPattern": 0.2,
        "CreditUtilization": 0.1
    }
    high_risk_input = {
        "MonthlyIncome": 500,
        "AccountBalance": 200,
        "SpendingPattern": 0.9,
        "CreditUtilization": 0.95
    }
    print("Low Risk Input Prediction:", predict(model, low_risk_input))
    print("High Risk Input Prediction:", predict(model, high_risk_input))

    # Explain prediction
    explain_prediction(model, sample_input)

if __name__ == "__main__":
    main()
