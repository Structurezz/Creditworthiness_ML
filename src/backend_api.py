from flask import Flask, request, jsonify
import os
import joblib
import openai
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import certifi
from src.data_preprocessing import validate_and_preprocess_input
from src.prediction_service import load_model, call_openai_api
from src.model_training import train_model
from src.utils import ensure_directory
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Load the trained model
model_path = "models/credit_model.pkl"  # Ensure this path is correct
model = None

def normalize_score(raw_score, raw_min=0, raw_max=100, min_score=300, max_score=850):
    # Normalize the score to fit within the credit score range
    return round(min_score + ((raw_score - raw_min) / (raw_max - raw_min)) * (max_score - min_score))

def calculate_risk_level(normalized_score):
    # Categorize risk level based on normalized credit score
    if normalized_score >= 750:
        return "Low Risk"
    elif normalized_score >= 700:
        return "Medium Risk"
    elif normalized_score >= 550:
        return "High Risk"
    else:
        return "Bad Risk"

# Example Data
ai_raw_score = 36.298
normalized_credit_score = normalize_score(ai_raw_score)
risk_level = calculate_risk_level(normalized_credit_score)

print(f"Normalized Credit Score: {normalized_credit_score}, Risk Level: {risk_level}")



def calculate_credit_score_and_risk(processed_data):
    # Extract data
    monthly_income = processed_data.get("MonthlyIncome", 0)
    account_balance = processed_data.get("AccountBalance", 0)
    credit_utilization = processed_data.get("CreditUtilization", 0)  # Percent
    spending_pattern = processed_data.get("SpendingPattern", 0)  # Percent

    # Initialize credit score and risk level
    credit_score = 850
    risk_level = "Low Risk"

    # Adjustments for credit utilization
    if credit_utilization > 50:
        credit_score -= 120  # Heavier penalty for very high utilization
        risk_level = "High Risk"
    elif credit_utilization > 30:
        credit_score -= 70  # Moderate penalty
        risk_level = "Medium Risk"

    # Adjustments for low income
    if monthly_income < 2000:
        credit_score -= 60  # Penalize for low income
        if credit_score < 650:  # Add higher risk if score drops below 650
            risk_level = "Medium Risk"
        else:
            risk_level = "Medium Risk"

    # Adjustments for low account balance
    if account_balance < 1000:
        credit_score -= 70  # Penalize for very low balance
        risk_level = "High Risk"

    # Adjustments for spending pattern
    if spending_pattern > 90:
        credit_score -= 190  # Heavy penalty for excessive spending
        risk_level = "High Risk"
    elif spending_pattern > 40:
        credit_score -= 80  # Moderate penalty for high spending
        if risk_level != "High Risk":  # Don't downgrade if already high
            risk_level = "Medium Risk"

    # Ensure credit score stays within bounds
    credit_score = max(300, min(credit_score, 850))

    return credit_score, risk_level





def save_to_database(data, risk_level, credit_score):
    try:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI is not set in environment variables.")

        client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = client["creditworthiness"]
        collection = db["predictions"]

        document = {
            "data": data,
            "risk_level": risk_level,
            "credit_score": credit_score
        }

        result = collection.insert_one(document)
        print(f"Prediction saved to database with ID: {result.inserted_id}")

    except (ValueError, PyMongoError) as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            client.close()
        except NameError:
            pass

        

@app.route("/train", methods=["POST"])
def train_model_api():
    global model_path
    try:
        # Retrieve data_folder and model_path from the request
        data_folder = request.json.get("data_folder", "data/raw/")
        model_path = request.json.get("model_path", "models/credit_model.pkl")

        # Debugging: Log the types and values of the variables
        print(f"Received data_folder: {data_folder} (type: {type(data_folder)})") 
        print(f"Received model_path: {model_path} (type: {type(model_path)})")
        print(f"Attempting to list files in: {data_folder} (type: {type(data_folder)})")
        
        # Ensure data_folder is a string before proceeding
        if not isinstance(data_folder, str):
            return jsonify({"error": "data_folder should be a string path."}), 400

        # Ensure the data_folder exists
        if not os.path.exists(data_folder):
            return jsonify({"error": f"Data folder '{data_folder}' does not exist."}), 404

        # List all CSV files in the given folder
        try:
            all_files = os.listdir(data_folder)
            print(f"All files in {data_folder}: {all_files}")  # Debugging output
            csv_files = [f for f in all_files if f.endswith('.csv')]
        except Exception as e:
            return jsonify({"error": f"Error listing files in {data_folder}: {str(e)}"}), 500

        # Check if there are no CSV files
        if not csv_files:
            return jsonify({"error": "No CSV files found in the specified data folder."}), 404

        ai_recommendations = []
        # Process each CSV file individually
        for csv_file in csv_files:
            file_path = os.path.join(data_folder, csv_file)
            print(f"Processing file: {file_path}")

            # Load the individual dataset
            data = pd.read_csv(file_path)

            if data.empty:
                print(f"No valid data found in {csv_file}.")
                continue  # Skip empty datasets

            print(f"Dataset shape for {csv_file}: {data.shape}")
            print(data.head())

            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                return jsonify({"error": "Loaded data is not a valid DataFrame."}), 500

            # Get AI recommendations for the current dataset
          

        # Load the last trained model
        global model
        model = load_model(model_path)

        return jsonify({
            "message": "Model trained and saved successfully.",
            "model_path": model_path,
            "ai_recommendations": ai_recommendations
        })
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Log error message
        return jsonify({"error": str(e)}), 500




@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    try:
        # Validate and preprocess the input data
        processed_data = validate_and_preprocess_input(data)

        # Calculate credit score and risk level based on the model
        credit_score, risk_level = calculate_credit_score_and_risk(processed_data)

        # Log calculated values for verification
        print(f"Calculated Credit Score: {credit_score}, Risk Level: {risk_level}")

        # Call OpenAI API for AI insights
        ai_insight = call_openai_api(
            f"Based on the following data: {data}, provide insights on the credit score using number and risk level."
        )

        # Log AI insight for verification
        print(f"AI Insight: {ai_insight}")

        # Optionally extract and compare AI insights
        if ai_insight:
            # Extract credit score from AI insight using regex
            credit_score_match = re.search(r'credit score\s*:\s*(\d{1,3})', ai_insight, re.IGNORECASE)
            if credit_score_match:
                ai_credit_score = int(credit_score_match.group(1))
                print(f"AI Recommended Credit Score: {ai_credit_score}")
                # Adjust calculated score based on AI recommendations if needed
                credit_score = adjust_credit_score(credit_score, ai_credit_score)

            # Extract risk level from AI insight
            risk_level_match = re.search(r'risk level\s*:\s*(\w+)', ai_insight, re.IGNORECASE)
            if risk_level_match:
                risk_level_text = risk_level_match.group(1).lower()
                risk_level = adjust_risk_level(risk_level, risk_level_text)

        # Save to database
        save_to_database(processed_data, risk_level, credit_score)

        # Return the results as JSON
        return jsonify({
            "credit_score": credit_score,
            "risk_level": risk_level,
            "ai_insight": ai_insight.strip() if ai_insight else None
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def adjust_credit_score(calculated_score, ai_score):
    # Adjust calculated score slightly towards AI score
    return int((calculated_score + ai_score) / 2)  # Average as an example

def adjust_risk_level(current_risk, ai_risk):
    # Adjust risk level based on AI insights
    if ai_risk != current_risk:
        # Logic to prefer the AI risk level if it's higher
        if ai_risk == "high":
            return "High Risk"
        elif ai_risk == "modarte" and current_risk != "High Risk":
            return "Medium Risk"
    return current_risk


if __name__ == "__main__":
    app.run(debug=True)
