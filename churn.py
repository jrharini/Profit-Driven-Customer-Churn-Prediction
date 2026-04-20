import pandas as pd

# Load dataset
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID
data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric (fix critical issue)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Convert Churn column to binary
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical variables using one-hot encoding (only categorical columns)
categorical_cols = data.select_dtypes(include=["object"]).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

print(data.head())

# Save preprocessed data for modeling
data.to_csv("preprocessed_churn.csv", index=False)

print("Preprocessing completed and data saved.")