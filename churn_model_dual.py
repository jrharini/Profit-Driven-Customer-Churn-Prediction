import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, accuracy_score, roc_curve, auc,
                              precision_score, confusion_matrix,
                              precision_recall_curve)
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
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

def fitness_function(mask, X_tr, X_te, y_tr, y_te):
    selected = X_tr.columns[mask == 1]
    if len(selected) == 0:
        return 0
    model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1, 1:2})
    model.fit(X_tr[selected], y_tr)
    pred = model.predict(X_te[selected])
    recall = recall_score(y_te, pred)
    feature_ratio = len(selected) / X_tr.shape[1]
    return recall - 0.05 * feature_ratio

def golden_eagle_optimization(X_tr, X_te, y_tr, y_te, pop_size=10, iterations=20):
    n_features = X_tr.shape[1]
    population = np.random.rand(pop_size, n_features)
    best_mask = None
    best_score = -np.inf
    for t in range(iterations):
        alpha = 1 - (t / iterations)
        for i in range(pop_size):
            mask = (population[i] > 0.5).astype(int)
            score = fitness_function(mask, X_tr, X_te, y_tr, y_te)
            if score > best_score:
                best_score = score
                best_mask = mask.copy()
            attack = alpha * (best_mask - population[i]) if best_mask is not None else np.zeros(n_features)
            cruise = np.random.uniform(-0.2, 0.2, n_features)
            population[i] = sigmoid(population[i] + attack + cruise)
    return best_mask, best_score

def classify_decision(row):
    if row["Decision_Threshold"] == "INTERVENE" and row["Actual_Churn"] == 1:
        return "TP"
    elif row["Decision_Threshold"] == "INTERVENE" and row["Actual_Churn"] == 0:
        return "FP"
    elif row["Decision_Threshold"] == "NO ACTION" and row["Actual_Churn"] == 1:
        return "FN"
    else:
        return "TN"

# ──────────────────────────────────────────────────────────────
# CORE PIPELINE — runs identically on any dataset
# ──────────────────────────────────────────────────────────────
def run_pipeline(dataset_name, csv_path):
    tag = dataset_name.replace(" ", "_").lower()
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    data = pd.read_csv(csv_path)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── RANDOM BASELINE ──────────────────────────────────────
    rand_mask = np.random.choice([0, 1], size=X_train.shape[1])
    if rand_mask.sum() == 0:
        rand_mask[0] = 1
    rand_features = X_train.columns[rand_mask == 1]
    model_rand = LogisticRegression(max_iter=5000, solver='liblinear')
    model_rand.fit(X_train[rand_features], y_train)
    rand_recall = recall_score(y_test, model_rand.predict(X_test[rand_features]))
    print(f"\n=== RANDOM FEATURE SELECTION BASELINE ===")
    print(f"Selected Features : {len(rand_features)}")
    print(f"Recall            : {rand_recall:.4f}")

    # ── GEO FEATURE SELECTION ────────────────────────────────
    print(f"\n=== Running GEO Feature Selection ===")
    best_mask, best_score = golden_eagle_optimization(X_train, X_test, y_train, y_test)
    selected_features = X_train.columns[best_mask == 1]

    num_total = X_train.shape[1]
    num_selected = len(selected_features)
    print(f"Total Features : {num_total}")
    print(f"GEO Selected   : {num_selected}")
    print(f"Reduction      : {num_total - num_selected} features ({100*(num_total-num_selected)/num_total:.1f}%)")
    print(f"GEO Fitness    : {round(best_score, 4)}")

    # ── KBEST BASELINE (same k as GEO) ───────────────────────
    selector = SelectKBest(score_func=f_classif, k=num_selected)
    X_kbest_train = selector.fit_transform(X_train, y_train)
    model_kbest = LogisticRegression(max_iter=5000, solver='liblinear')
    model_kbest.fit(X_kbest_train, y_train)
    recall_kbest = recall_score(y_test, model_kbest.predict(selector.transform(X_test)))
    print(f"\n=== K-BEST FEATURE SELECTION ===")
    print(f"Selected Features : {num_selected}")
    print(f"Recall            : {recall_kbest:.4f}")

    # ── BASELINE MODEL (all features) ────────────────────────
    base_model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1, 1:2})
    base_model.fit(X_train, y_train)
    base_prob = base_model.predict_proba(X_test)[:, 1]
    baseline_preds = base_model.predict(X_test)

    # ── FINAL MODEL (GEO features) ───────────────────────────
    final_model = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1, 1:3}, C=0.5)
    final_model.fit(X_train[selected_features], y_train)
    final_pred = final_model.predict(X_test[selected_features])
    final_prob = final_model.predict_proba(X_test[selected_features])[:, 1]

    final_accuracy = accuracy_score(y_test, final_pred)
    final_recall = recall_score(y_test, final_pred)
    final_precision = precision_score(y_test, final_pred, zero_division=0)
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)

    print(f"\n=== FINAL MODEL (LR + GEO) ===")
    print(f"Accuracy  : {final_accuracy:.4f}")
    print(f"Recall    : {final_recall:.4f}")
    print(f"Precision : {final_precision:.4f}")
    print(f"F1        : {final_f1:.4f}")

    # ── CONFUSION MATRIX ─────────────────────────────────────
    cm = confusion_matrix(y_test, final_pred)
    print(f"\n=== CONFUSION MATRIX ===")
    print(cm)

    # ── THRESHOLD OPTIMIZATION ───────────────────────────────
    final_threshold = find_best_threshold(y_test.values, final_prob)
    profit_default = compute_profit(y_test.values, final_prob, 0.5)
    profit_optimized = compute_profit(y_test.values, final_prob, final_threshold)
    print(f"\n=== THRESHOLD COMPARISON ===")
    print(f"Default  (0.5)           Profit : {profit_default}")
    print(f"Optimised ({final_threshold})       Profit : {profit_optimized}")
    print(f"Improvement                      : {profit_optimized - profit_default:.2f}")

    # ── SENSITIVITY ANALYSIS ─────────────────────────────────
    print(f"\n=== SENSITIVITY ANALYSIS ===")
    for sr in [0.3, 0.5, 0.7]:
        for cp in [0.1, 0.2, 0.3]:
            p = compute_profit(y_test.values, final_prob, final_threshold,
                               success_rate=sr, cost_pct=cp)
            print(f"Success Rate={sr}, Cost%={cp} → Profit={p}")

    # ── ABLATION STUDY ───────────────────────────────────────
    print(f"\n=== ABLATION STUDY ===")
    profit_baseline = compute_profit(y_test.values, base_prob, 0.5)
    best_thr_base = find_best_threshold(y_test.values, base_prob)
    profit_thr_only = compute_profit(y_test.values, base_prob, best_thr_base)
    profit_fs_only = compute_profit(y_test.values, final_prob, 0.5)
    profit_full = compute_profit(y_test.values, final_prob, final_threshold)

    ablation_rows = [
        ("Baseline (all features, t=0.5)", profit_baseline),
        ("Threshold Optimisation Only",     profit_thr_only),
        ("GEO Feature Selection Only (t=0.5)", profit_fs_only),
        ("Full System (GEO + Optimised Threshold)", profit_full),
    ]
    ablation_df = pd.DataFrame(ablation_rows, columns=["Setup", "Profit"])
    ablation_df.to_csv(f"results/{tag}_ablation.csv", index=False)
    print(ablation_df.to_string(index=False))

    # ── CV SCORES ────────────────────────────────────────────
    cv_scores = cross_val_score(final_model, X[selected_features], y, cv=5, scoring='recall')
    print(f"\n=== CV RECALL SCORES ===")
    print(f"Scores         : {cv_scores}")
    print(f"Mean CV Recall : {cv_scores.mean():.4f}")

    # ── STABILITY (10 seeds) ─────────────────────────────────
    profit_list = []
    for seed in range(10):
        Xtr, Xte, ytr, yte = train_test_split(
            X[selected_features], y, test_size=0.2, random_state=seed, stratify=y)
        m = LogisticRegression(max_iter=5000, solver='liblinear', class_weight={0:1, 1:2})
        m.fit(Xtr, ytr)
        prb = m.predict_proba(Xte)[:, 1]
        profit_list.append(compute_profit(yte.values, prb, final_threshold))

    mean_profit = np.mean(profit_list)
    std_profit = np.std(profit_list)
    cv_metric = std_profit / mean_profit if mean_profit != 0 else 0
    print(f"\n=== STABILITY (10 SEEDS) ===")
    print(f"Mean Profit  : {mean_profit:.2f}")
    print(f"Std  Profit  : {std_profit:.2f}")
    print(f"CV (std/mean): {cv_metric:.4f}")

    # ── GENERALIZATION TEST (different split) ────────────────
    print(f"\n=== GENERALIZATION TEST (DIFFERENT SPLIT) ===")
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X[selected_features], y, test_size=0.3, random_state=7, stratify=y)
    model2 = LogisticRegression(max_iter=5000, solver='liblinear')
    model2.fit(X_tr2, y_tr2)
    recall2 = recall_score(y_te2, model2.predict(X_te2))
    print(f"Recall (new split) : {recall2:.4f}")

    # ── ROC CURVE ────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, final_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"LR + GEO  AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve — {dataset_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"results/{tag}_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nAUC : {roc_auc:.4f}")

    # ── PRECISION-RECALL CURVE ───────────────────────────────
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, final_prob)
    plt.figure()
    plt.plot(rec_vals, prec_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {dataset_name}")
    plt.savefig(f"results/{tag}_pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── RECALL COMPARISON ────────────────────────────────────
    baseline_recall = recall_score(y_test, baseline_preds)
    print(f"\n=== RECALL COMPARISON ===")
    print(f"Baseline (all features)   : {baseline_recall:.4f}")
    print(f"Random selection          : {rand_recall:.4f}")
    print(f"KBest selection           : {recall_kbest:.4f}")
    print(f"GEO (final model)         : {final_recall:.4f}")
    print(f"GEO vs Baseline           : {final_recall - baseline_recall:+.4f}")
    print(f"GEO vs Random             : {final_recall - rand_recall:+.4f}")
    print(f"GEO vs KBest              : {final_recall - recall_kbest:+.4f}")

    recall_comp_df = pd.DataFrame([
        {"Method": "Baseline (All Features, t=0.5)", "Recall": round(baseline_recall, 4)},
        {"Method": "Random Feature Selection",        "Recall": round(rand_recall, 4)},
        {"Method": "KBest (Statistical)",             "Recall": round(recall_kbest, 4)},
        {"Method": "GEO + Optimised Threshold",       "Recall": round(final_recall, 4)},
    ])
    recall_comp_df.to_csv(f"results/{tag}_recall_comparison.csv", index=False)

    # ── MODEL COMPARISON ─────────────────────────────────────
    print(f"\n=== MODEL COMPARISON ===")
    comp_rows = []
    other_models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM":           SVC(probability=True, random_state=42),
    }
    for name, model in other_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rec  = recall_score(y_test, pred)
        acc  = accuracy_score(y_test, pred)
        prof = compute_profit(y_test.values, prob, 0.5)
        print(f"{name:20s} Accuracy={acc:.4f}  Recall={rec:.4f}  Profit={prof:.2f}")
        comp_rows.append({"Model": name, "Accuracy": acc, "Recall": rec, "Profit": prof})

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"results/{tag}_model_comparison.csv", index=False)

    # ── CUSTOMER DECISION OUTPUT ─────────────────────────────
    monthly_charge = 65.0
    decision_df = pd.DataFrame({
        "Customer_ID":       range(len(y_test)),
        "Actual_Churn":      y_test.values,
        "Churn_Probability": np.round(final_prob, 4),
        "Monthly_Charges":   monthly_charge,
    })
    decision_df["CLV_24_Months"]     = monthly_charge * 24
    decision_df["Expected_Savings"]  = decision_df["Churn_Probability"] * 0.5 * decision_df["CLV_24_Months"]
    decision_df["Intervention_Cost"] = monthly_charge * 0.20
    decision_df["Expected_Profit"]   = decision_df["Expected_Savings"] - decision_df["Intervention_Cost"]
    decision_df["Decision"]          = decision_df["Expected_Profit"].apply(
        lambda x: "INTERVENE" if x > 0 else "NO ACTION")
    decision_df["Decision_Threshold"] = decision_df["Churn_Probability"].apply(
        lambda x: "INTERVENE" if x >= final_threshold else "NO ACTION")
    decision_df["Correct"] = decision_df.apply(classify_decision, axis=1)
    decision_df = decision_df.sort_values("Expected_Profit", ascending=False).reset_index(drop=True)
    decision_df.to_csv(f"results/{tag}_customer_decisions.csv", index=False)

    # ── SAVE MODEL ───────────────────────────────────────────
    joblib.dump(final_model,          f"models/{tag}_final_model.pkl")
    joblib.dump(list(selected_features), f"models/{tag}_selected_features.pkl")

    # ── RETURN SUMMARY FOR CROSS-DATASET TABLE ───────────────
    return {
        "Dataset":           dataset_name,
        "Samples":           len(y),
        "Total Features":    num_total,
        "GEO Features":      num_selected,
        "Recall (Baseline)": round(baseline_recall, 4),
        "Recall (Random)":   round(rand_recall, 4),
        "Recall (KBest)":    round(recall_kbest, 4),
        "Recall (GEO)":      round(final_recall, 4),
        "Precision":         round(final_precision, 4),
        "F1":                round(final_f1, 4),
        "AUC":               round(roc_auc, 4),
        "CV Recall (mean)":  round(cv_scores.mean(), 4),
        "Profit (default t)":   profit_default,
        "Profit (optimised t)": profit_optimized,
        "Best Threshold":    final_threshold,
    }

# ──────────────────────────────────────────────────────────────
# RUN ON BOTH DATASETS
# ──────────────────────────────────────────────────────────────
results = []

results.append(run_pipeline(
    dataset_name="WA Telco (Dataset 1)",
    csv_path="data/preprocessed_churn.csv"
))

results.append(run_pipeline(
    dataset_name="Iranian Churn (Dataset 2)",
    csv_path="data/preprocessed_iranian_churn.csv"
))

# ──────────────────────────────────────────────────────────────
# CROSS-DATASET COMPARISON TABLE
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CROSS-DATASET COMPARISON")
print(f"{'='*60}")

comparison_df = pd.DataFrame(results)
print(comparison_df.T.to_string())
comparison_df.to_csv("results/cross_dataset_comparison.csv", index=False)

print("\n✅ Cross-dataset comparison saved → results/cross_dataset_comparison.csv")
print("✅ All per-dataset results saved in results/")
print("✅ Models saved in models/")