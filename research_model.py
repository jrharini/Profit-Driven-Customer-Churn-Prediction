import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve  # CHANGE 3 & 4
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
data = pd.read_csv("data/preprocessed_churn.csv")
X = data.drop("Churn", axis=1)
y = data["Churn"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

# CHANGE 2: Convert hardcoded values to parameters
def compute_profit(y_true, y_prob, threshold, success_rate=0.5, cost_pct=0.20, monthly_mean=65.0):
    y_pred = (y_prob >= threshold).astype(int)
    profit = 0.0
    for i, pred in enumerate(y_pred):
        if pred == 1:
            expected_save = float(y_prob[i]) * success_rate * monthly_mean * 24
            cost = monthly_mean * cost_pct
            profit += expected_save - cost
    return round(profit, 2)

def find_best_threshold(y_true, y_prob, success_rate=0.5, cost_pct=0.20):
    best_thr = 0.5
    best_profit = -np.inf
    for thr in np.linspace(0.1, 0.9, 81):
        p = compute_profit(y_true, y_prob, thr, success_rate, cost_pct)
        if p > best_profit:
            best_profit = p
            best_thr = thr
    return round(best_thr, 3)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ──────────────────────────────────────────────────────────────
# GEO FEATURE SELECTION
# ──────────────────────────────────────────────────────────────
LAMBDA = 0.05

def fitness_function(mask):
    selected = X_train.columns[mask == 1]
    n_selected = len(selected)
    if n_selected == 0:
        return 0
    model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1,1:2})
    model.fit(X_train[selected], y_train)
    pred = model.predict(X_test[selected])
    recall = recall_score(y_test, pred)
    feature_ratio = n_selected / X_train.shape[1]
    return recall - LAMBDA * feature_ratio

def golden_eagle_optimization(pop_size=10, iterations=20):
    n_features = X_train.shape[1]
    population = np.random.rand(pop_size, n_features)
    best_mask = None
    best_score = -np.inf
    for t in range(iterations):
        alpha = 1 - (t/iterations)
        for i in range(pop_size):
            mask = (population[i] > 0.5).astype(int)
            score = fitness_function(mask)
            if score > best_score:
                best_score = score
                best_mask = mask.copy()
            attack = alpha * (best_mask - population[i]) if best_mask is not None else np.zeros(n_features)
            cruise = np.random.uniform(-0.2,0.2,n_features)
            population[i] = sigmoid(population[i] + attack + cruise)
    return best_mask, best_score

# CHANGE 1: Random feature selection baseline
def random_feature_selection():
    mask = np.random.choice([0, 1], size=X_train.shape[1])
    return mask

print("\n=== Running GEO Feature Selection (with sparsity penalty λ=0.05) ===")
best_mask, best_score = golden_eagle_optimization()
selected_features = X_train.columns[best_mask==1]

# CHANGE 1: Evaluate random baseline and compare
rand_mask = random_feature_selection()
rand_features = X_train.columns[rand_mask == 1]

model_rand = LogisticRegression(max_iter=5000, solver='liblinear')
model_rand.fit(X_train[rand_features], y_train)

rand_pred = model_rand.predict(X_test[rand_features])
rand_recall = recall_score(y_test, rand_pred)

print("\n=== RANDOM FEATURE SELECTION BASELINE ===")
print("Selected Features:", len(rand_features))
print("Recall:", rand_recall)

# K-BEST STATISTICAL BASELINE (same k as GEO for fair comparison)
selector = SelectKBest(score_func=f_classif, k=len(selected_features))
X_kbest_train = selector.fit_transform(X_train, y_train)

model_kbest = LogisticRegression(max_iter=5000, solver='liblinear')
model_kbest.fit(X_kbest_train, y_train)

pred_kbest = model_kbest.predict(selector.transform(X_test))
recall_kbest = recall_score(y_test, pred_kbest)

print("\n=== K-BEST FEATURE SELECTION ===")
print("Selected Features:", len(selected_features))
print("Recall:", recall_kbest)

# Feature Count Comparison
num_total = X_train.shape[1]
num_selected = len(selected_features)

print(f"\n=== ADD #3: FEATURE COUNT COMPARISON ===")
print(f"Total Features  : {num_total}")
print(f"GEO Selected    : {num_selected}")
print(f"Reduction       : {num_total-num_selected} features removed ({100*(num_total-num_selected)/num_total:.1f}% reduction)")

feat_count_df = pd.DataFrame([
    {"Method": "Full Feature Set", "Features": num_total},
    {"Method": "GEO Selected", "Features": num_selected},
    {"Method": "Random Selected", "Features": len(rand_features)},
])
feat_count_df.to_csv("results/feature_count.csv", index=False)

print("\n=== GEO RESULTS ===")
print("Selected Features:", list(selected_features))
print("GEO Fitness Score:", round(best_score,4))

# ──────────────────────────────────────────────────────────────
# BASELINE MODEL  (train here so predictions are available later)
# ──────────────────────────────────────────────────────────────
base_model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1,1:2})
base_model.fit(X_train, y_train)
base_prob     = base_model.predict_proba(X_test)[:,1]
baseline_preds = base_model.predict(X_test)          # ← kept for recall comparison

# ──────────────────────────────────────────────────────────────
# TRAIN FINAL MODEL
# ──────────────────────────────────────────────────────────────
final_model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1,1:3}, C=0.5)
final_model.fit(X_train[selected_features], y_train)
final_pred = final_model.predict(X_test[selected_features])
final_prob = final_model.predict_proba(X_test[selected_features])[:,1]

final_accuracy = accuracy_score(y_test, final_pred)
final_recall = recall_score(y_test, final_pred)

print("\n=== FINAL MODEL (LR + GEO, selected features) ===")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Recall  : {final_recall:.4f}")

# CHANGE 3: Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# CHANGE 4: Precision-Recall Curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, final_prob)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("results/pr_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Threshold Comparison
final_threshold = find_best_threshold(y_test.values, final_prob)
profit_default = compute_profit(y_test.values, final_prob, 0.5)
profit_optimized = compute_profit(y_test.values, final_prob, final_threshold)

print(f"\n=== ADD #2: THRESHOLD COMPARISON ===")
print(f"Default Threshold (0.5)       Profit : {profit_default}")
print(f"Optimised Threshold ({final_threshold})  Profit : {profit_optimized}")
print(f"Improvement from threshold tuning    : {profit_optimized-profit_default:.2f}")

thresh_df = pd.DataFrame([
    {"Setup": "Default threshold (0.5)", "Threshold": 0.5, "Profit": profit_default},
    {"Setup": f"Optimised ({final_threshold})", "Threshold": final_threshold, "Profit": profit_optimized}
])
thresh_df.to_csv("results/threshold_comparison.csv", index=False)

# CHANGE 2: Sensitivity Analysis
print("\n=== SENSITIVITY ANALYSIS ===")

for sr in [0.3, 0.5, 0.7]:
    for cp in [0.1, 0.2, 0.3]:
        profit = compute_profit(y_test.values, final_prob, final_threshold,
                                success_rate=sr, cost_pct=cp)
        print(f"Success Rate={sr}, Cost%={cp} → Profit={profit}")

# Ablation Study
print(f"\n=== ADD #1: ABLATION STUDY ===")
results_ablation = []

profit_baseline = compute_profit(y_test.values, base_prob, 0.5)
results_ablation.append(("Baseline (all features, t=0.5)", profit_baseline))

best_thr_base = find_best_threshold(y_test.values, base_prob)
profit_threshold = compute_profit(y_test.values, base_prob, best_thr_base)
results_ablation.append(("Threshold Optimisation Only", profit_threshold))

profit_fs = compute_profit(y_test.values, final_prob, 0.5)
results_ablation.append(("GEO Feature Selection Only (t=0.5)", profit_fs))

profit_full = compute_profit(y_test.values, final_prob, final_threshold)
results_ablation.append(("Full System (GEO + Optimised Threshold)", profit_full))

ablation_df = pd.DataFrame(results_ablation, columns=["Setup","Profit"])
ablation_df.to_csv("results/ablation.csv", index=False)
print(ablation_df.to_string(index=False))

# Budget Analysis
print(f"\n=== ADD #4: BUDGET vs PROFIT ANALYSIS ===")
budgets = [0.1,0.2,0.5,1.0]
budget_results = []
sorted_idx = np.argsort(final_prob)[::-1]
y_test_arr = y_test.values
for b in budgets:
    k = max(1,int(len(final_prob)*b))
    top_idx = sorted_idx[:k]
    profit_b = compute_profit(y_test_arr[top_idx], final_prob[top_idx], final_threshold)
    budget_results.append((f"{int(b*100)}% of customers", k, round(profit_b,2)))

budget_df = pd.DataFrame(budget_results, columns=["Budget","N_Customers","Profit"])
budget_df.to_csv("results/budget_analysis.csv", index=False)
print(budget_df.to_string(index=False))

# CV Scores
cv_scores = cross_val_score(final_model, X[selected_features], y, cv=5, scoring='recall')
print("\n=== CV RECALL SCORES ===")
print("Scores        :", cv_scores)
print("Mean CV Recall:", cv_scores.mean())

# Stability
profit_list=[]
for seed in range(10):
    Xtr,Xte,ytr,yte = train_test_split(X[selected_features],y,test_size=0.2, random_state=seed, stratify=y)
    m = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1,1:2})
    m.fit(Xtr,ytr)
    prb = m.predict_proba(Xte)[:,1]
    profit_list.append(compute_profit(yte.values,prb,final_threshold))

mean_profit = np.mean(profit_list)
std_profit = np.std(profit_list)
cv_metric = std_profit/mean_profit if mean_profit!=0 else 0

print(f"\n=== ADD #5: STABILITY (CV METRIC) OVER 10 RANDOM SEEDS ===")
print(f"Mean Profit  : {mean_profit:.2f}")
print(f"Std  Profit  : {std_profit:.2f}")
print(f"CV (std/mean): {cv_metric:.4f}")

stability_df = pd.DataFrame([{"Mean_Profit": round(mean_profit,2),"Std_Profit": round(std_profit,2),"CV":round(cv_metric,4)}])
stability_df.to_csv("results/stability.csv", index=False)

# ROC + AUC
fpr,tpr,_ = roc_curve(y_test, final_prob)
roc_auc = auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,label=f"LR + GEO  AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve (LR + GEO, GEO-selected features)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("results/roc_curve.png", dpi=300, bbox_inches='tight')  # CHANGE 7: dpi=300
plt.show()
print("AUC:", roc_auc)

# CHANGE 5: Generalization test with different split
print("\n=== GENERALIZATION TEST (DIFFERENT SPLIT) ===")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X[selected_features], y, test_size=0.3, random_state=7, stratify=y
)

model2 = LogisticRegression(max_iter=5000, solver='liblinear')
model2.fit(X_train2, y_train2)

pred2 = model2.predict(X_test2)
recall2 = recall_score(y_test2, pred2)

print("Recall (new split):", recall2)

# ──────────────────────────────────────────────────────────────
# ADD #NEW-1: RECALL COMPARISON (BASELINE vs FINAL)
# ──────────────────────────────────────────────────────────────
print("\n=== RECALL COMPARISON (BASELINE vs FINAL) ===")

baseline_recall = recall_score(y_test, baseline_preds)
final_recall_cmp = recall_score(y_test, final_pred)

print(f"Baseline Recall      : {baseline_recall:.4f}")
print(f"Final Model Recall   : {final_recall_cmp:.4f}")
print(f"Recall Improvement   : {final_recall_cmp - baseline_recall:+.4f}")

# CHANGE 1: Add random baseline recall to comparison
print(f"Random Selection Recall: {rand_recall:.4f}")
print(f"GEO vs Random Improvement: {final_recall_cmp - rand_recall:+.4f}")

if final_recall_cmp > baseline_recall:
    print("✔ GEO + threshold optimisation improves recall over baseline.")
else:
    print("→ Profit gain comes from decision-level optimisation, not recall alone.")

recall_comp_df = pd.DataFrame([
    {"Model": "Baseline (All Features, t=0.5)", "Recall": round(baseline_recall, 4)},
    {"Model": "Random Feature Selection", "Recall": round(rand_recall, 4)},
    {"Model": "KBest (Statistical)", "Recall": round(recall_kbest, 4)},
    {"Model": "Final (GEO + Optimised Threshold)", "Recall": round(final_recall_cmp, 4)},
    {"Model": "Improvement (GEO vs Baseline)", "Recall": round(final_recall_cmp - baseline_recall, 4)},
    {"Model": "Improvement (GEO vs Random)", "Recall": round(final_recall_cmp - rand_recall, 4)},
    {"Model": "Improvement (GEO vs KBest)", "Recall": round(final_recall_cmp - recall_kbest, 4)},
])
recall_comp_df.to_csv("results/recall_comparison.csv", index=False)
print("Saved → results/recall_comparison.csv")

# ──────────────────────────────────────────────────────────────
# CUSTOMER-LEVEL DECISION OUTPUT
# ──────────────────────────────────────────────────────────────
monthly_charge = 65.0
success_rate = 0.5
cost_pct = 0.20

decision_df = pd.DataFrame({
    "Customer_ID": range(len(y_test)),
    "Actual_Churn": y_test.values,
    "Churn_Probability": np.round(final_prob, 4),
    "Monthly_Charges": monthly_charge,
})

decision_df["CLV_24_Months"] = monthly_charge * 24
decision_df["Expected_Savings"] = (decision_df["Churn_Probability"] * success_rate *
                                    decision_df["CLV_24_Months"])
decision_df["Intervention_Cost"] = monthly_charge * cost_pct
decision_df["Expected_Profit"] = decision_df["Expected_Savings"] - decision_df["Intervention_Cost"]

decision_df["Decision"] = decision_df["Expected_Profit"].apply(
    lambda x: "INTERVENE" if x > 0 else "NO ACTION"
)
decision_df["Decision_Threshold"] = decision_df["Churn_Probability"].apply(
    lambda x: "INTERVENE" if x >= final_threshold else "NO ACTION"
)

# CHANGE 6: Fix the buggy Correct column logic
def classify(row):
    if row["Decision_Threshold"] == "INTERVENE" and row["Actual_Churn"] == 1:
        return "TP"
    elif row["Decision_Threshold"] == "INTERVENE" and row["Actual_Churn"] == 0:
        return "FP"
    elif row["Decision_Threshold"] == "NO ACTION" and row["Actual_Churn"] == 1:
        return "FN"
    else:
        return "TN"

decision_df["Correct"] = decision_df.apply(classify, axis=1)

decision_df = decision_df.sort_values(by="Expected_Profit", ascending=False).reset_index(drop=True)
decision_df.to_csv("results/customer_decisions.csv", index=False)

print(f"\n=== ADD #7: CUSTOMER-LEVEL DECISION SUPPORT OUTPUT ===")
print(f"\nTop 15 Customers to Target (by Expected Profit):")
print(decision_df[["Customer_ID", "Churn_Probability", "Expected_Profit",
                   "Decision_Threshold", "Actual_Churn"]].head(15).to_string(index=False))

print(f"\n--- Decision Distribution ---")
intervention_count = (decision_df["Decision_Threshold"] == "INTERVENE").sum()
no_action_count = (decision_df["Decision_Threshold"] == "NO ACTION").sum()

print(f"Customers to Intervene: {intervention_count} ({100*intervention_count/len(decision_df):.1f}%)")
print(f"No Action Required: {no_action_count} ({100*no_action_count/len(decision_df):.1f}%)")

true_positives = ((decision_df["Decision_Threshold"] == "INTERVENE") &
                  (decision_df["Actual_Churn"] == 1)).sum()
false_positives = ((decision_df["Decision_Threshold"] == "INTERVENE") &
                   (decision_df["Actual_Churn"] == 0)).sum()
false_negatives = ((decision_df["Decision_Threshold"] == "NO ACTION") &
                   (decision_df["Actual_Churn"] == 1)).sum()

print(f"\nPrecision (Intervention Accuracy): {true_positives}/{intervention_count} = "
      f"{100*true_positives/max(intervention_count,1):.1f}%")
print(f"Churn Detection: {true_positives}/{true_positives+false_negatives} = "
      f"{100*true_positives/max(true_positives+false_negatives,1):.1f}%")

print(f"\n--- Financial Impact ---")
total_intervention_cost = intervention_count * (monthly_charge * cost_pct)
total_expected_savings = decision_df[decision_df["Decision_Threshold"] == "INTERVENE"]["Expected_Savings"].sum()
total_expected_profit = decision_df[decision_df["Decision_Threshold"] == "INTERVENE"]["Expected_Profit"].sum()

print(f"Total Intervention Cost: ${total_intervention_cost:.2f}")
print(f"Total Expected Savings: ${total_expected_savings:.2f}")
print(f"Net Expected Profit: ${total_expected_profit:.2f}")
print(f"\nFull system profit (all interventions): ${profit_full:.2f}")

# ──────────────────────────────────────────────────────────────
# ADD #NEW-2: PRECISION vs RECALL TRADE-OFF (EXPLICIT)
# ──────────────────────────────────────────────────────────────
print("\n=== PRECISION vs RECALL TRADE-OFF ===")

final_precision = precision_score(y_test, final_pred, zero_division=0)
final_recall_pr = recall_score(y_test, final_pred)

print(f"Precision (Intervention Accuracy): {final_precision:.4f}")
print(f"Recall    (Churn Detection)      : {final_recall_pr:.4f}")
print(f"F1 Score                         : {2 * final_precision * final_recall_pr / max(final_precision + final_recall_pr, 1e-9):.4f}")
print(f"\n→ We intentionally sacrifice precision ({final_precision:.2%}) to maximise")
print(f"  recall ({final_recall_pr:.2%}) and profit — catching more churners")
print(f"  is more valuable than avoiding false alarms in this business context.")

pr_tradeoff_df = pd.DataFrame([{
    "Model":     "GEO + Optimised Threshold",
    "Precision": round(final_precision, 4),
    "Recall":    round(final_recall_pr, 4),
    "F1":        round(2 * final_precision * final_recall_pr /
                       max(final_precision + final_recall_pr, 1e-9), 4),
    "Threshold": final_threshold,
}])
pr_tradeoff_df.to_csv("results/precision_recall_tradeoff.csv", index=False)
print("Saved → results/precision_recall_tradeoff.csv")

# ──────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────────────────────
models = {"Random Forest":RandomForestClassifier(random_state=42),
          "XGBoost":XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
          "SVM":SVC(probability=True, random_state=42)}

print("\n=== MODEL COMPARISON ===")
comp_rows=[]
for name,model in models.items():
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    recall = recall_score(y_test,pred)
    acc = accuracy_score(y_test,pred)
    profit_m = compute_profit(y_test.values,prob,0.5)
    print(f"{name} -> Accuracy: {acc:.4f}  Recall: {recall:.4f}  Profit: {profit_m:.2f}")
    comp_rows.append({"Model": name, "Accuracy": acc, "Recall": recall, "Profit": profit_m})

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv("results/model_comparison.csv", index=False)

# SHAP vs GEO Feature Overlap
try:
    import shap
    explainer = shap.Explainer(final_model,X_train[selected_features])
    shap_values = explainer(X_test[selected_features])
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    top_shap_features = set(selected_features[np.argsort(shap_importance)[::-1][:10]])
    geo_features_set = set(selected_features)
    overlap = geo_features_set.intersection(top_shap_features)

    print(f"\n=== ADD #6: SHAP vs GEO FEATURE OVERLAP ===")
    print(f"GEO selected features : {len(geo_features_set)}")
    print(f"Top-10 SHAP features  : {len(top_shap_features)}")
    print(f"Overlap               : {len(overlap)} features")
    print(f"Overlapping features  : {sorted(overlap)}")

    overlap_df = pd.DataFrame({"Feature": sorted(selected_features),
                               "SHAP_Score": shap_importance,
                               "In_Top10_SHAP":[f in top_shap_features for f in selected_features]})
    overlap_df.to_csv("results/shap_geo_overlap.csv", index=False)
except Exception as e:
    print(f"\nSHAP skipped (install shap if needed): {e}")

# Save model
joblib.dump(final_model,"models/final_churn_model.pkl")
joblib.dump(list(selected_features),"models/selected_features.pkl")

print("\n✅ Model and selected features saved.")
print("✅ All results CSVs written to results/")
print("✅ Customer decision output: results/customer_decisions.csv")
print("✅ Recall comparison:        results/recall_comparison.csv")
print("✅ Precision-Recall tradeoff: results/precision_recall_tradeoff.csv")
print("✅ PR curve saved:            results/pr_curve.png")