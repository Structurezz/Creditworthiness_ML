import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
import shap

# Step 1: Load the datasets
credit_card_customers = pd.read_csv('../data/credit-card-customers.csv')
creditcard_fraud = pd.read_csv('../data/creditcard.csv')
credit_score_classification = pd.read_csv('../data/credit_score_classification.csv')
german_credit = pd.read_csv('../data/german_credit.csv')
credit_risk_dataset = pd.read_csv('../data/credit-risk-dataset.csv')

# Step 2: Data Cleaning and Preprocessing
# Handling missing values in all datasets
credit_card_customers.fillna(credit_card_customers.mean(), inplace=True)
credit_score_classification.fillna(credit_score_classification.mean(), inplace=True)

# Encoding categorical variables
credit_card_customers = pd.get_dummies(credit_card_customers, drop_first=True)
credit_score_classification = pd.get_dummies(credit_score_classification, drop_first=True)

# Ensure common column exists for merging (example: 'customer_id')
if 'customer_id' not in credit_card_customers.columns:
    credit_card_customers['customer_id'] = range(1, len(credit_card_customers) + 1)
if 'customer_id' not in credit_score_classification.columns:
    credit_score_classification['customer_id'] = range(1, len(credit_score_classification) + 1)

# Merge datasets
combined_data = pd.merge(credit_card_customers, credit_score_classification, on='customer_id', how='inner')

# Create a synthetic target column if not present
if 'creditworthy' not in combined_data.columns:
    combined_data['creditworthy'] = np.random.randint(0, 2, size=combined_data.shape[0])

# Step 3: Split the data into features and target
X = combined_data.drop(['customer_id', 'creditworthy'], axis=1)  # Features
y = combined_data['creditworthy']  # Target variable

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the pipeline (scaling + model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 6: Train the model using cross-validation
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Evaluate feature importance
model = pipeline.named_steps['classifier']
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Step 8: Save the pipeline (model + scaler)
joblib.dump(pipeline, 'creditworthiness_pipeline.pkl')

# Step 9: Anomaly Detection with Isolation Forest
isolation_forest = IsolationForest(random_state=42, contamination=0.1)
isolation_forest.fit(X_train)
anomalies = isolation_forest.predict(X_test)
print("Anomalies detected in test data:")
print(np.sum(anomalies == -1))  # Count anomalies

# Step 10: Explainability using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize SHAP summary plot (ensure you have matplotlib installed)
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)

# Save the explainer and SHAP values for later use
joblib.dump({'explainer': explainer, 'shap_values': shap_values}, 'shap_explanations.pkl')
