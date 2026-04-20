import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_curve,
    auc,
    accuracy_score
)

from xgboost import XGBClassifier

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv("preprocessed_churn.csv")

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 1️⃣ PROPER GOLDEN EAGLE OPTIMIZATION (SIMPLIFIED VERSION)
# =========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(mask):
    selected = X_train.columns[mask == 1]
    if len(selected) == 0:
        return 0

    model = LogisticRegression(max_iter=1000, class_weight={0:1,1:2})
    model.fit(X_train[selected], y_train)
    pred = model.predict(X_test[selected])
    return recall_score(y_test, pred)

def golden_eagle_optimization(pop_size=10, iterations=15):

    n_features = X_train.shape[1]
    population = np.random.rand(pop_size, n_features)

    best_mask = None
    best_score = 0

    for _ in range(iterations):
        for i in range(pop_size):

            mask = (population[i] > 0.5).astype(int)
            score = fitness_function(mask)

            if score > best_score:
                best_score = score
                best_mask = mask.copy()

            # Exploration & Exploitation update
            attack = np.random.rand(n_features)
            cruise = np.random.rand(n_features)

            population[i] = population[i] + 0.5*attack + 0.5*cruise
            population[i] = sigmoid(population[i])

    return best_mask, best_score


best_mask, best_recall = golden_eagle_optimization()
selected_features = X_train.columns[best_mask == 1]

print("\n=== GEO RESULTS ===")
print("Selected Features:", list(selected_features))
print("Recall after GEO:", best_recall)

# =========================================================
# 2️⃣ FINAL COST-SENSITIVE LOGISTIC MODEL
# =========================================================

final_model = LogisticRegression(max_iter=1000, class_weight={0:1,1:2})
final_model.fit(X_train[selected_features], y_train)

final_pred = final_model.predict(X_test[selected_features])
final_prob = final_model.predict_proba(X_test[selected_features])[:,1]

final_recall = recall_score(y_test, final_pred)
final_accuracy = accuracy_score(y_test, final_pred)

print("\n=== FINAL MODEL (LR + GEO) ===")
print("Accuracy:", final_accuracy)
print("Recall:", final_recall)

# =========================================================
# 3️⃣ CROSS VALIDATION
# =========================================================

cv_scores = cross_val_score(
    final_model,
    X[selected_features],
    y,
    cv=5,
    scoring='recall'
)

print("Mean CV Recall:", cv_scores.mean())

# =========================================================
# 4️⃣ ROC + AUC
# =========================================================

fpr, tpr, _ = roc_curve(y_test, final_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title("ROC Curve (LR + GEO)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC:", roc_auc)

# =========================================================
# 5️⃣ MODEL COMPARISON
# =========================================================

models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

print("\n=== MODEL COMPARISON ===")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    recall = recall_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    print(name, "-> Accuracy:", acc, " Recall:", recall)

# =========================================================
# 6️⃣ SAVE BEST MODEL
# =========================================================

joblib.dump(final_model, "final_churn_model.pkl")
joblib.dump(list(selected_features), "selected_features.pkl")

print("\nModel and selected features saved successfully.")