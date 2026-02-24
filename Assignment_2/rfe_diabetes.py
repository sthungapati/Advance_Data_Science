import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.01

def main():
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    print("Dataset shape:", X.shape)
    print("Features:", feature_names)
    print("Target: disease progression (continuous)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Baseline
    baseline = LinearRegression().fit(X_train, y_train)
    baseline_r2 = r2_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline (k=10) Test R2: {baseline_r2:.6f}")

    results_rows = []
    # coefficient table: dict k -> dict(feature -> coef)
    coef_by_k = {}

    for k in range(10, 0, -1):
        selector = RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
        selector.fit(X_train, y_train)

        support = selector.support_  # boolean mask
        selected_features = [f for f, keep in zip(feature_names, support) if keep]
        sel_idx = np.where(support)[0]

        model_k = LinearRegression().fit(X_train[:, sel_idx], y_train)
        r2 = r2_score(y_test, model_k.predict(X_test[:, sel_idx]))

        coef_map = {f: c for f, c in zip(selected_features, model_k.coef_)}
        coef_by_k[k] = coef_map

        results_rows.append({
            "k": k,
            "r2": r2,
            "selected_features": ", ".join(selected_features)
        })

        print(f"k={k} -> Test R2={r2:.6f} | features={selected_features}")

    # Sort results by k descending (10 -> 1)
    results_rows.sort(key=lambda d: d["k"], reverse=True)

    # Choose optimal k by threshold: smallest k where removing one more drops >= THRESHOLD
    opt_k = None
    for i in range(len(results_rows) - 1):
        r2_now = results_rows[i]["r2"]
        r2_next = results_rows[i + 1]["r2"]
        if (r2_now - r2_next) >= THRESHOLD:
            opt_k = results_rows[i]["k"]
            break
    if opt_k is None:
        opt_k = max(results_rows, key=lambda d: d["r2"])["k"]

    print(f"\nChosen optimal k (threshold={THRESHOLD}): {opt_k}")
    print("Selected features:", results_rows[[r["k"] for r in results_rows].index(opt_k)]["selected_features"])

    # Save CSV: rfe_r2_results.csv
    with open("rfe_r2_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "r2", "selected_features"])
        w.writeheader()
        for row in results_rows:
            w.writerow(row)

    # Save coefficient-by-iteration CSV: rfe_coefficients_by_iteration.csv
    # rows = k_retained, columns = features, blank if eliminated
    with open("rfe_coefficients_by_iteration.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["k_retained"] + feature_names
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k in range(10, 0, -1):
            row = {"k_retained": k}
            for feat in feature_names:
                row[feat] = coef_by_k[k].get(feat, "")
            w.writerow(row)

    # Plot R2 vs k
    ks = [row["k"] for row in results_rows][::-1]      # 1..10
    r2s = [row["r2"] for row in results_rows][::-1]    # aligned

    plt.figure()
    plt.plot(ks, r2s, marker="o")
    plt.xlabel("Number of retained features (k)")
    plt.ylabel("Test R2 score")
    plt.title("RFE with Linear Regression on Diabetes Dataset")
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    plt.savefig("r2_vs_num_features.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()