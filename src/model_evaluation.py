# src/model_evaluation.py
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(y_true, y_pred):
    """Evaluate the model using classification metrics."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")

def cross_validate_model(model, X, y, cv=5):
    """Perform k-fold cross-validation on the model."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(scores):.2f}")
