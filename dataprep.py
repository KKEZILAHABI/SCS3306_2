import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib

# 1. Load dataset
df = pd.read_csv('carclaims.csv')

# 2. Handle missing values (zeros) in specified columns
missing_value_cols = ['Age', 'DayOfWeekClaimed', 'MonthClaimed']
for col in missing_value_cols:
    df[col] = df[col].replace(0, np.nan)

# 3. Drop rows with missing values in the specified columns
df = df.dropna(subset=missing_value_cols)

# 4. Define columns FIRST to establish a strict ordering schema
numeric_cols = ['WeekOfMonth', 'WeekOfMonthClaimed', 'Age', 'RepNumber', 'Deductible', 'DriverRating', 'Year', 'Deductible']
categorical_cols = [col for col in df.columns if col not in numeric_cols + ['FraudFound_P']]

# 5. ENFORCE column ordering (Numeric -> Categorical -> Target) to prevent misalignment in prediction
feature_cols = numeric_cols + categorical_cols
df = df[feature_cols + ['FraudFound_P']]

# 6. Clean categorical data AFTER defining the final categorical_cols list
for col in categorical_cols:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        # Strip whitespace and convert to lowercase for consistency
        df[col] = df[col].astype(str).str.lower().str.strip()

# 7. Split features/target
X = df.drop(columns=['FraudFound_P'])
y = df['FraudFound_P']

# 8. Train/Val/Test split (Adding .copy() to safely mutate dataframes)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

# 9. Ordinal encode categoricals
encoders = {}
for col in categorical_cols:
    # Get all possible values from the full dataset to ensure complete encoding
    all_values = df[col].dropna().unique().tolist()
    
    # Create encoder with all known categories to avoid unknown values later
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', 
                        unknown_value=-1, 
                        categories=[all_values])
                        
    # FIX: Use direct assignment instead of .loc to allow Pandas to change the column dtype from string to float
    X_train[col] = enc.fit_transform(X_train[[col]])
    X_val[col] = enc.transform(X_val[[col]])
    X_test[col] = enc.transform(X_test[[col]])
    encoders[col] = enc
    
    # Extra debugging - check if any values got encoded as -1
    unknown_vals = X_train[X_train[col] == -1][col].unique()
    if len(unknown_vals) > 0:
        print(f"WARNING: Found unknown values in {col} during training: {unknown_vals}")

# 10. Save artifacts
joblib.dump(encoders, "encoders.joblib")
joblib.dump(categorical_cols, "categorical_cols.joblib")
joblib.dump(numeric_cols, "numeric_cols.joblib")

# Save categories for each encoder
categories_dict = {col: enc.categories_[0].tolist() for col, enc in encoders.items()}
joblib.dump(categories_dict, "categories.joblib")

# 11. Encoding report
with open("encoding_report.txt", "w") as f:
    f.write("Encoding Report\n====================\n")
    f.write(f"Missing values (0) handled for: {', '.join(missing_value_cols)}\n\n")
    f.write("Categorical preprocessing: Lowercase and whitespace trimming applied\n\n")
    for col in categorical_cols:
        categories = encoders[col].categories_[0]
        f.write(f"{col}: OrdinalEncoder (unknown -> -1)\n")
        f.write(f"  Known categories ({len(categories)}): {categories[:10]}")
        if len(categories) > 10:
            f.write(f"... plus {len(categories) - 10} more\n")
        else:
            f.write("\n")

# 12. Save datasets with target
X_train['FraudFound_P'] = y_train.values
X_val['FraudFound_P'] = y_val.values
X_test['FraudFound_P'] = y_test.values

X_train.to_csv("train.csv", index=False)
X_val.to_csv("validation.csv", index=False)
X_test.to_csv("test.csv", index=False)

print("Data preparation complete. Artifacts saved.")