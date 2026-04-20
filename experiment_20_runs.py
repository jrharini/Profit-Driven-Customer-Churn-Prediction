import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.stats import ttest_rel


# ==============================
# GLOBAL SETTINGS
# ==============================
LAMBDA = 0.01
np.random.seed(42)


# ==============================
# LOAD DATASET
# ==============================
data = pd.read_csv("data/preprocessed_churn.csv")

X = data.drop("Churn", axis=1)
y = data["Churn"]


# ==============================
# JACCARD STABILITY FUNCTION
# ==============================
def compute_stability(feature_masks):
    similarities = []
    n = len(feature_masks)

    for i in range(n):
        for j in range(i + 1, n):
            set1 = set(np.where(feature_masks[i] == 1)[0])
            set2 = set(np.where(feature_masks[j] == 1)[0])

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            if union == 0:
                similarities.append(1)
            else:
                similarities.append(intersection / union)

    return np.mean(similarities)


# ==============================
# GEO FITNESS FUNCTION
# ==============================
def fitness_function(mask, X_train, y_train):

    selected_indices = np.where(mask == 1)[0]

    if len(selected_indices) == 0:
        return 0

    X_selected = X_train[:, selected_indices]

    model = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1, 1: 2},
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    auc_scores = []

    for train_idx, val_idx in cv.split(X_selected, y_train):
        X_tr = X_selected[train_idx]
        X_val = X_selected[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, prob))

    mean_auc = np.mean(auc_scores)
    feature_ratio = len(selected_indices) / X_train.shape[1]

    return mean_auc - LAMBDA * feature_ratio


# ==============================
# GOLDEN EAGLE OPTIMIZATION
# ==============================
def golden_eagle_optimization(X_train, y_train, pop_size=8, iterations=15, mutation_rate=0.05):

    n_features = X_train.shape[1]

    population = np.random.randint(0, 2, (pop_size, n_features))
    fitness = np.array([fitness_function(ind, X_train, y_train) for ind in population])

    best_idx = np.argmax(fitness)
    best_mask = population[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(iterations):

        alpha = 1 - (t / iterations)

        for i in range(pop_size):

            attack = alpha * (best_mask - population[i])
            cruise = np.random.uniform(-0.2, 0.2, n_features)

            new_position = population[i] + attack + cruise

            sigmoid = 1 / (1 + np.exp(-new_position))
            new_mask = (np.random.rand(n_features) < sigmoid).astype(int)

            mutation = np.random.rand(n_features) < mutation_rate
            new_mask = np.logical_xor(new_mask, mutation).astype(int)

            new_score = fitness_function(new_mask, X_train, y_train)

            if new_score > fitness[i]:
                population[i] = new_mask
                fitness[i] = new_score

                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask.copy()

    return best_mask


# ==============================
# PROPOSED METHOD (CONSENSUS)
# ==============================
def run_proposed_method(X_train, X_test, y_train, y_test):

    masks = []
    for _ in range(5):
        mask = golden_eagle_optimization(X_train, y_train)
        masks.append(mask)

    masks = np.array(masks)

    vote_counts = np.sum(masks, axis=0)
    consensus_mask = (vote_counts >= 3).astype(int)

    if np.sum(consensus_mask) == 0:
        consensus_mask = masks[0]

    selected_indices = np.where(consensus_mask == 1)[0]

    X_train_final = X_train[:, selected_indices]
    X_test_final = X_test[:, selected_indices]

    model = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1, 1: 2},
        random_state=42
    )

    model.fit(X_train_final, y_train)

    y_prob = model.predict_proba(X_test_final)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, consensus_mask


# ==============================
# L1 FEATURE SELECTION
# ==============================
def run_l1_feature_selection(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    mask = (model.coef_[0] != 0).astype(int)
    selected = np.where(mask == 1)[0]

    X_train_final = X_train[:, selected]
    X_test_final = X_test[:, selected]

    model2 = LogisticRegression(max_iter=1000, random_state=42)
    model2.fit(X_train_final, y_train)

    y_prob = model2.predict_proba(X_test_final)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, mask


# ==============================
# SELECTKBEST FEATURE SELECTION
# ==============================
def run_selectkbest(X_train, X_test, y_train, y_test):

    k = int(X_train.shape[1] * 0.6)

    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)

    mask = selector.get_support().astype(int)
    selected = np.where(mask == 1)[0]

    X_train_final = X_train[:, selected]
    X_test_final = X_test[:, selected]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_final, y_train)

    y_prob = model.predict_proba(X_test_final)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return auc_score, mask


# ==============================
# EXPERIMENT LOOP
# ==============================
def run_feature_selection_experiment(method_function, name):

    auc_scores = []
    runtimes = []
    masks = []

    print(f"\nRunning {name} for 20 independent runs...")

    for i in range(20):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        start = time.time()
        auc, mask = method_function(X_train, X_test, y_train, y_test)
        end = time.time()

        auc_scores.append(auc)
        runtimes.append(end - start)
        masks.append(mask)

        print(f"Run {i+1}: AUC={auc:.4f}")

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_time = np.mean(runtimes)
    stability = compute_stability(masks)
    avg_features = np.mean([np.sum(m) for m in masks])

    print(f"\n{name} Results:")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Avg Runtime: {mean_time:.2f} sec")
    print(f"Average Selected Features: {avg_features:.2f}")
    print(f"Stability (Jaccard): {stability:.4f}")

    return auc_scores


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    auc_geo = run_feature_selection_experiment(run_proposed_method, "Proposed GEO (Consensus)")
    auc_l1 = run_feature_selection_experiment(run_l1_feature_selection, "L1 Logistic")
    auc_kbest = run_feature_selection_experiment(run_selectkbest, "SelectKBest")

    t_stat, p_value = ttest_rel(auc_geo, auc_l1)

    print("\nPaired t-test (GEO vs L1)")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)