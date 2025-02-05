# tests/test_creditworthiness.py
import unittest
from src.data_preparation import load_data, preprocess_data
from src.model_training import train_model
from src.prediction_service import load_model, predict

class TestCreditworthiness(unittest.TestCase):

    def setUp(self):
        self.raw_data_path = "data/raw/employee_attrition.csv"
        self.processed_data_path = "data/processed/creditworthiness_data.csv"
        self.model_path = "models/credit_model.pkl"
        
    def test_load_data(self):
        data = load_data(self.raw_data_path)
        self.assertFalse(data.empty, "Loaded data should not be empty")

    def test_preprocess_data(self):
        raw_data = load_data(self.raw_data_path)
        processed_data = preprocess_data(raw_data)
        self.assertIn("Creditworthiness", processed_data.columns, "Processed data should contain 'Creditworthiness' column")

    def test_train_model(self):
        train_model(self.processed_data_path, self.model_path)

    def test_predict(self):
        model = load_model(self.model_path)
        sample_input = {
            "MonthlyIncome": 4500,
            "AccountBalance": 8000,
            "SpendingPattern": 0.6,
            "CreditUtilization": 0.4
        }
        prediction = predict(model, sample_input)
        self.assertIn(prediction, ["Low Risk", "Medium Risk", "High Risk"], "Prediction should be one of the risk levels")

if __name__ == "__main__":
    unittest.main()
