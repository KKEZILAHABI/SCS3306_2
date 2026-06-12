import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             precision_recall_curve, auc, roc_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import StandardScaler

# Load data and artifacts
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")
numeric_cols = joblib.load("numeric_cols.joblib")  

# Split features/target
target_column = 'FraudFound_P'
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_val = val_df.drop(columns=[target_column])
y_val = val_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Initialize scaler and scale ONLY numeric columns
scaler = StandardScaler()

# Scale numeric columns while preserving original structure
X_train_scaled = X_train.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

X_val_scaled = X_val.copy()
X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

X_test_scaled = X_test.copy()
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Convert to numpy arrays
X_train_scaled = X_train_scaled.values
X_val_scaled = X_val_scaled.values
X_test_scaled = X_test_scaled.values

# Model training
clf = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
param_dist = {
    'max_iter': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_leaf': [20, 30, 50]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(clf, param_dist, scoring='f1', cv=cv, n_iter=20, 
                            verbose=2, n_jobs=-1, random_state=42)
search.fit(X_train_scaled, y_train)
best_model = search.best_estimator_

# Calibration
calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method='sigmoid')
calibrated_model.fit(X_val_scaled, y_val)

# Threshold tuning
val_probs = calibrated_model.predict_proba(X_val_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

# FIX: Slice precision and recall to match the thresholds array length 
f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Test evaluation
test_probs = calibrated_model.predict_proba(X_test_scaled)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_test, test_preds)
precision_score_val = precision_score(y_test, test_preds, zero_division=0)
recall_score_val = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probs)
pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probs)
pr_auc = auc(pr_recall, pr_precision)
conf_matrix = confusion_matrix(y_test, test_preds)
class_report = classification_report(y_test, test_preds)

# Save artifacts
joblib.dump(calibrated_model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
with open("optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

# Save performance report
with open("performance.txt", "w") as f:
    f.write(f"""
Improved Model Performance Report

Best Parameters:
{search.best_params_}

Validation Optimal Threshold: {optimal_threshold:.4f}

Test Set Metrics:
-----------------------
Accuracy: {accuracy:.4f}
Precision: {precision_score_val:.4f}
Recall: {recall_score_val:.4f}
F1 Score: {f1:.4f}
ROC AUC: {roc_auc:.4f}
PR AUC: {pr_auc:.4f}

Confusion Matrix:
{conf_matrix}

Classification Report:
{class_report}
""")

# Generate plots
fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

plt.figure()
plt.plot(pr_recall, pr_precision, label=f"PR (AUC = {pr_auc:.4f})")
plt.axhline(y_test.mean(), color='r', linestyle='--', label='Baseline')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("pr_curve.png")
plt.close()

print("Training complete. All artifacts saved.")