from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import ssl
import glob

# Load environment variables from .env file
load_dotenv()

model_path = "models/credit_model.pkl"  # Default model path
model = None 

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic data for creditworthiness."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=4, 
        n_classes=3,
        n_informative=4, 
        n_redundant=0, 
        weights=[0.5, 0.3, 0.2],
        random_state=42
    )
    score_mapping = {0: 300, 1: 600, 2: 850}
    numerical_scores = [score_mapping[label] for label in y]
    synthetic_data = pd.DataFrame(X, columns=["MonthlyIncome", "AccountBalance", "SpendingPattern", "CreditUtilization"])
    synthetic_data["CreditScore"] = numerical_scores
    return synthetic_data

def inspect_class_distribution(data, column="CreditScore"):
    """Inspect the distribution of target classes in the dataset."""
    class_counts = data[column].value_counts()
    total_samples = len(data)
    print("Class Distribution:")
    print(class_counts)
    print("\nClass Distribution Percentage:")
    print((class_counts / total_samples) * 100)

def load_data_from_mongodb():
    """Load training data from MongoDB."""
    mongo_uri = os.getenv("MONGODB_URI")
    try:
        client = MongoClient(mongo_uri, ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
        db = client["creditworthiness"]
        collection = db["training_data"]
        data = pd.DataFrame(list(collection.find()))  # Ensure 'find()' is called correctly
        return data
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def load_data_from_csv_folder(folder_path):
    """Load and combine all CSV files from the specified folder into a single DataFrame."""
    dataframes = []

    # List all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)  # Read each CSV file into a DataFrame
                dataframes.append(df)  # Append the DataFrame to the list
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        combined_data = pd.concat(dataframes, ignore_index=True)
        return combined_data
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no valid CSV files were found


def train_model(folder_path: str, save_path: str):
    """Train a model to predict creditworthiness."""
    
    # Try loading data from CSV files in the folder
    data = load_data_from_csv_folder(folder_path)
    if data.empty:
        print("No data files found or all data files are empty, generating synthetic data.")
        data = generate_synthetic_data(n_samples=1000)

    # Inspect class distribution before training
    print("\nInspecting class distribution:")
    inspect_class_distribution(data)

    # Prepare features and target variable
    X = data[["MonthlyIncome", "AccountBalance", "SpendingPattern", "CreditUtilization"]]
    y = data["CreditScore"]

    # Ensure 'CreditScore' is in the DataFrame
    if 'CreditScore' not in data.columns:
        print("CreditScore column is missing from the data.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure enough data for training
    if len(X_train) < 2:
        print("Not enough samples in training data.")
        return

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error (MSE): {mse}")

    # Save the trained model
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    RAW_DATA_FOLDER = "data/raw/"  # Path to your raw data folder containing CSV files
    MODEL_PATH = "models/credit_model.pkl"
    train_model(RAW_DATA_FOLDER, MODEL_PATH)
