#!/usr/bin/env python3
"""
ml_threat_detector.py — Trains and evaluates two models on the UCI
Phishing Websites dataset (11,055 samples, 30 binary/ternary features,
label: 1 = legitimate, -1 = phishing):

  1. A supervised Random Forest Classifier (benign vs. malicious).
  2. An unsupervised Isolation Forest anomaly detector, treating the
     minority class (phishing) as the anomaly.

Run:
    python3 ml_threat_detector.py

Writes:
    results/classification_report.txt
    results/model_comparison.md
    results/class_distribution.txt
    results/head.txt
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "phishing_raw.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LABEL_COL = "Result"  # 1 = legitimate (benign), -1 = phishing (malicious)


def load_and_clean_data(path: str):
    df = pd.read_csv(path)

    head_str = df.head().to_string()
    class_dist = df[LABEL_COL].value_counts()
    class_dist_str = (
        f"{LABEL_COL} value counts (1 = legitimate/benign, -1 = phishing/malicious):\n"
        f"{class_dist.to_string()}\n\n"
        f"Proportions:\n{(class_dist / len(df) * 100).round(2).to_string()}%"
    )

    # Drop rows with nulls (dataset has none, but keep this defensive per
    # the task spec so the pipeline is robust to messier inputs).
    before_na = len(df)
    df = df.dropna()
    dropped_na = before_na - len(df)

    # All 30 feature columns in this dataset are already integer-encoded
    # (-1/0/1), so there is no free-text categorical column left to
    # one-hot/label-encode; this step is a no-op here but is included so
    # the pipeline generalizes to other labelled security datasets that
    # do carry string categoricals.
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)

    # Duplicate rows are common in this dataset because most features are
    # low-cardinality (-1/0/1), so many distinct websites end up with
    # identical feature vectors. We drop exact-duplicate rows and report
    # the count removed, per the task spec.
    before_dupes = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before_dupes - len(df)

    return df, head_str, class_dist_str, dropped_na, dropped_dupes


def train_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)  # default hyperparameters, per task spec
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["phishing(-1)", "legitimate(1)"])
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=-1),  # precision on the malicious class
        "recall": recall_score(y_test, y_pred, pos_label=-1),
        "f1": f1_score(y_test, y_pred, pos_label=-1),
    }
    return clf, report, metrics


def train_isolation_forest(X_train, X_test, y_test, contamination):
    # Isolation Forest is unsupervised: it never sees y_train. `contamination`
    # tells it what fraction of points to expect as anomalies, estimated
    # here from the training set's minority-class proportion.
    iso = IsolationForest(random_state=42, contamination=contamination)
    iso.fit(X_train)

    # IsolationForest.predict returns 1 for "normal"/inlier and -1 for
    # "anomaly"/outlier — which conveniently matches this dataset's own
    # 1 (legitimate) / -1 (phishing) label convention, so no remapping
    # is needed before comparing to y_test.
    y_pred = iso.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=-1),
        "recall": recall_score(y_test, y_pred, pos_label=-1),
        "f1": f1_score(y_test, y_pred, pos_label=-1),
    }
    return iso, metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, head_str, class_dist_str, dropped_na, dropped_dupes = load_and_clean_data(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Dropped {dropped_na} null row(s), {dropped_dupes} duplicate row(s)")

    with open(os.path.join(RESULTS_DIR, "head.txt"), "w") as f:
        f.write(head_str + "\n")
    with open(os.path.join(RESULTS_DIR, "class_distribution.txt"), "w") as f:
        f.write(class_dist_str + "\n")

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Supervised: Random Forest ---
    rf_model, rf_report, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    print("\n=== Random Forest classification report ===")
    print(rf_report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(rf_report)

    # --- Unsupervised: Isolation Forest ---
    # The task instructs us to treat the MINORITY class as the modeled
    # anomaly. In the original, pre-deduplication dataset, phishing (-1)
    # is the minority class (4,898 of 11,055 samples, 44.3%; see README
    # "Class distribution"), so we treat phishing as the anomaly class
    # throughout this script, including for the contamination estimate
    # below. Note: deduplication happens to leave the remaining samples
    # close to balanced (see README), so "minority" no longer applies in
    # a strict post-preprocessing sense — this is a deliberate design
    # choice based on the dataset's original class proportions and the
    # standard SOC convention of treating malicious activity as the
    # class of interest, not a literal recomputation after dedup.
    raw_contamination = (y_train == -1).mean()
    contamination = min(max(raw_contamination, 0.01), 0.5)
    iso_model, iso_metrics = train_isolation_forest(X_train, X_test, y_test, contamination)
    print("\n=== Isolation Forest anomaly-detection metrics ===")
    for k, v in iso_metrics.items():
        print(f"{k}: {v:.4f}")

    # --- Comparison table ---
    comparison = (
        "| Model | Accuracy | Precision | Recall | F1 Score | Notes |\n"
        "|---|---|---|---|---|---|\n"
        f"| Random Forest (supervised) | {rf_metrics['accuracy']:.4f} | {rf_metrics['precision']:.4f} "
        f"| {rf_metrics['recall']:.4f} | {rf_metrics['f1']:.4f} "
        "| Trained on labelled data; precision/recall computed for the phishing(-1) class |\n"
        f"| Isolation Forest (unsupervised) | {iso_metrics['accuracy']:.4f} | {iso_metrics['precision']:.4f} "
        f"| {iso_metrics['recall']:.4f} | {iso_metrics['f1']:.4f} "
        f"| Never saw labels during training; contamination parameter set to "
        f"{contamination:.4f} (train-set phishing proportion) |\n"
    )
    with open(os.path.join(RESULTS_DIR, "model_comparison.md"), "w") as f:
        f.write(comparison)
    print("\n=== Model comparison ===")
    print(comparison)


if __name__ == "__main__":
    main()
