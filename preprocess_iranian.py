import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/Customer_Churn.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename Churn to confirm it matches pipeline expectation
# Already named 'Churn' with values 0/1 — no changes needed

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Churn distribution:\n", df["Churn"].value_counts())
print("Nulls:", df.isnull().sum().sum())

df.to_csv("data/preprocessed_iranian_churn.csv", index=False)
print("\n✅ Saved → data/preprocessed_iranian_churn.csv")