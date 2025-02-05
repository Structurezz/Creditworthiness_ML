import joblib
import pandas as pd
import shap
import openai

def load_model(model_path: str):
    """Load a trained model."""
    return joblib.load(model_path)

def calculate_credit_score(input_data):
    """Calculate a credit score based on input data."""
    score = 0
    # Example scoring logic
    if input_data['MonthlyIncome'] > 5000:
        score += 30
    if input_data['AccountBalance'] > 5000:
        score += 30
    if input_data['SpendingPattern'] < 0.5:
        score += 20
    if input_data['CreditUtilization'] < 0.3:
        score += 20

    # Normalize the score to a scale of 0-100
    return min(score, 100)
def calculate_credit_score_and_risk(data):
    """Calculate credit score dynamically and determine risk level."""
    income = data.get("monthly_income", 0)
    balance = data.get("account_balance", 0)
    utilization = data.get("credit_utilization", 0)

    # Start with a base score
    credit_score = 300

    # High utilization penalty
    if utilization > 0.5:
        credit_score -= 100
    elif utilization <= 0.3:
        credit_score += 50  # Reward low utilization

    # Low balance-to-income penalty
    if balance / income < 0.1:
        credit_score -= 100
    elif balance / income >= 0.3:
        credit_score += 50  # Reward healthy balance

    # High income bonus
    if income > 1000000:
        credit_score += 100

    # Cap credit score between 300 and 850
    credit_score = max(300, min(850, credit_score))

    # Determine risk level
    if credit_score >= 750:
        risk_level = "Low Risk"
    elif 500 <= credit_score < 750:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return credit_score, risk_level

def generate_ai_insight(income, balance, spending_pattern, credit_utilization, credit_score):
    risk_factors = []
    if income < 2000:
        risk_factors.append("low monthly income")
    if balance < income * 0.3:
        risk_factors.append("low account balance")
    if spending_pattern == "High":
        risk_factors.append("high spending pattern")
    if credit_utilization > 30:
        risk_factors.append("high credit utilization")

    risk_summary = " and ".join(risk_factors) if risk_factors else "a balanced financial profile"
    insight = f"This person demonstrates {risk_summary}."
    return insight

def predict(model, input_data):
    """Predict creditworthiness for a single input."""
    print("Input Data:", input_data)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    print("Input DataFrame:", input_df)
    print("DataFrame Types:", input_df.dtypes)

    try:
        # Predict using the model
        prediction = model.predict(input_df)
        print("Prediction Output:", prediction)

        # Interpret the prediction
        if isinstance(prediction[0], float):  # Regression output
            credit_score = round(prediction[0])  # Round to integer
        else:
            credit_score = prediction[0]  # Classification output

        # Map credit score to risk level
        if credit_score >= 700:
            risk_level = "Low Risk"
        elif credit_score >= 500:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        ai_insight = f"Predicted credit score: {credit_score}. Risk level: {risk_level}."
        return credit_score, risk_level, ai_insight
    except Exception as e:
        print(f"Error in model prediction: {e}")
        raise

def explain_prediction(model, input_data: dict):
    """Explain the prediction using SHAP."""
    explainer = shap.TreeExplainer(model)
    df = pd.DataFrame([input_data])
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values, df)

# Example usage
if __name__ == "__main__":
    # Load the model
    model_path = "models/credit_model.pkl"
    model = load_model(model_path)

    # Example input data
    example_input = {
        'MonthlyIncome': 1000,
        'AccountBalance': 390,
        'SpendingPattern': "High",
        'CreditUtilization': 50
    }

    # Predict and generate insights
    credit_score, risk_level, ai_insight, probabilities = predict(model, example_input)
    print(f"Credit Score: {credit_score}")
    print(f"Risk Level: {risk_level}")
    print(f"AI Insight: {ai_insight}")

def call_openai_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Failed to generate AI insights: {str(e)}"
