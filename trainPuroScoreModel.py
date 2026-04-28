"""
trainPuroScoreModel.py
─────────────────────
Trains a Logistic Regression model on allergen_dataset_v6_scored.csv
to predict whether a product is unsafe for a given allergen.

Input CSV columns (one row per product × allergen):
  product_name, allergen,
  has_direct_allergen, has_derived_allergen, has_contains_statement,
  has_may_contain, has_shared_facility, ambiguity_count,
  missing_data_flag, ocr_confidence,
  puro_score, confidence, triggers, label

Usage:
  python3 trainPuroScoreModel.py
  python3 trainPuroScoreModel.py --csv path/to/file.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ── Config ────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "has_direct_allergen",
    "has_derived_allergen",
    "has_contains_statement",
    "has_may_contain",
    "has_shared_facility",
    "ambiguity_count",
    "missing_data_flag",
    "ocr_confidence",
]

TARGET_COL  = "label"
MODEL_PATH  = "models/puroScoreModel_v1.joblib"
RANDOM_SEED = 42

# Per spec: recall is priority — use class_weight="balanced" + low threshold
DECISION_THRESHOLD = 0.30

ALLERGENS = [
    "Milk", "Eggs", "Fish", "Crustacean",
    "TreeNuts", "Peanuts", "Wheat", "Soybeans", "Sesame",
]

# Hardcoded v1 deductions for coefficient comparison (per spec Section 6)
V1_DEDUCTIONS = {
    "has_direct_allergen":    85,
    "has_derived_allergen":   75,
    "has_contains_statement": 85,
    "has_may_contain":        45,
    "has_shared_facility":    35,
    "ambiguity_count":        10,   # per ambiguous term
    "missing_data_flag":      50,
    "ocr_confidence":         -18,  # low confidence = higher risk (negative)
}

W = 65   # print width


def hr(c="─"): return c * W
def banner(title): return f"\n{'═'*W}\n  {title}\n{'═'*W}"


# ── 1. Load & Prep ────────────────────────────────────────────────────────────

def load_and_prep(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    print(banner("1. LOAD & PREP"))

    df = pd.read_csv(csv_path)
    print(f"  Loaded       : {csv_path}")
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.columns.tolist()}")

    # Validate expected columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"\n  ✗ Missing columns: {missing}")
        sys.exit(1)

    # Convert label → binary: unsafe=1, safe/uncertain=0
    def to_binary(val):
        if isinstance(val, (int, float)):
            return int(val)
        v = str(val).strip().lower()
        return 1 if v in ("unsafe", "1", "true", "yes") else 0

    df["label_binary"] = df[TARGET_COL].apply(to_binary)

    print(f"\n  Label distribution:")
    counts = df["label_binary"].value_counts()
    total  = len(df)
    for v, n in counts.sort_index().items():
        label = "unsafe (1)" if v == 1 else "safe    (0)"
        print(f"    {label} : {n:>5,}  ({n/total*100:.1f}%)")

    # Drop rows with missing feature values
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        print(f"\n  Dropped {before - len(df)} rows with missing feature values.")

    X = df[FEATURE_COLS].astype(float)
    y = df["label_binary"]

    print(f"\n  Features used: {FEATURE_COLS}")
    return X, y, df


# ── 2. Train / Test Split ─────────────────────────────────────────────────────

def split(X, y):
    print(banner("2. TRAIN / TEST SPLIT  (80 / 20)"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train rows : {len(X_train):,}")
    print(f"  Test rows  : {len(X_test):,}")
    print(f"  Unsafe in train : {y_train.sum():,}  ({y_train.mean()*100:.1f}%)")
    print(f"  Unsafe in test  : {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train(X_train, y_train) -> Pipeline:
    print(banner("3. TRAIN — Logistic Regression (recall-optimised)"))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="saga",
            class_weight="balanced",   # handles class imbalance, boosts recall
            random_state=RANDOM_SEED,
        )),
    ])
    pipe.fit(X_train, y_train)
    print(f"  ✓ Model trained  |  Threshold: {DECISION_THRESHOLD} (recall-optimised)")
    return pipe


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(pipe: Pipeline, X_test, y_test, df_full: pd.DataFrame, X: pd.DataFrame):
    print(banner("4. EVALUATE"))

    # Apply recall-optimised threshold
    probs    = pipe.predict_proba(X_test)[:, 1]
    y_pred   = (probs >= DECISION_THRESHOLD).astype(int)

    recall   = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  Overall (threshold = {DECISION_THRESHOLD})")
    print(f"  {hr()}")
    print(f"  Recall (sensitivity)   : {recall:.4f}   ← priority metric")
    print(f"  Precision              : {precision:.4f}")
    print(f"  False Negative Rate    : {fnr:.4f}   ← missed unsafe products")
    print(f"  True  Positives        : {tp:>5}")
    print(f"  False Negatives        : {fn:>5}   ← products we said safe but aren't")
    print(f"  False Positives        : {fp:>5}   ← safe products flagged as unsafe")
    print(f"  True  Negatives        : {tn:>5}")

    # ── Per-allergen breakdown ────────────────────────────────────────────────
    if "allergen" in df_full.columns:
        print(f"\n  Per-Allergen Breakdown")
        print(f"  {hr()}")
        print(f"  {'Allergen':<18} {'Recall':>7} {'Prec':>7} {'FNR':>7} "
              f"{'Unsafe':>7} {'Total':>7}")
        print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        test_indices = y_test.index
        df_test = df_full.loc[test_indices].copy()
        df_test["_pred"] = y_pred
        df_test["_prob"] = probs

        for allergen in ALLERGENS:
            mask = df_test["allergen"].str.lower() == allergen.lower()
            if mask.sum() == 0:
                continue
            sub_y    = df_test.loc[mask, "label_binary"]
            sub_pred = df_test.loc[mask, "_pred"]
            if sub_y.sum() == 0:
                rec = prec = fnr_a = float("nan")
            else:
                rec   = recall_score(sub_y, sub_pred, zero_division=0)
                prec  = precision_score(sub_y, sub_pred, zero_division=0)
                tn_a, fp_a, fn_a, tp_a = confusion_matrix(
                    sub_y, sub_pred, labels=[0, 1]
                ).ravel()
                fnr_a = fn_a / (fn_a + tp_a) if (fn_a + tp_a) > 0 else 0.0

            print(f"  {allergen:<18} {rec:>7.3f} {prec:>7.3f} {fnr_a:>7.3f} "
                  f"{sub_y.sum():>7} {len(sub_y):>7}")


# ── 5. Coefficients vs v1 Deductions ─────────────────────────────────────────

def print_coefficients(pipe: Pipeline):
    print(banner("5. LEARNED COEFFICIENTS vs v1 HARDCODED DEDUCTIONS"))

    lr_coef   = pipe.named_steps["lr"].coef_[0]
    intercept = pipe.named_steps["lr"].intercept_[0]

    # Normalise to same scale as v1 deductions for comparison
    max_coef = max(abs(c) for c in lr_coef) or 1.0
    max_ded  = max(abs(v) for v in V1_DEDUCTIONS.values()) or 1.0

    print(f"\n  {'Feature':<28} {'ML Coef':>9}  {'ML→85 scale':>12}  {'v1 Deduction':>13}  {'Δ':>8}")
    print(f"  {'─'*28} {'─'*9}  {'─'*12}  {'─'*13}  {'─'*8}")

    for feat, coef in zip(FEATURE_COLS, lr_coef):
        scaled_ml = (coef / max_coef) * max_ded
        v1_val    = V1_DEDUCTIONS.get(feat, 0)
        delta     = scaled_ml - v1_val
        direction = "↑ ML harsher" if delta > 5 else ("↓ ML softer" if delta < -5 else "≈ agree")
        print(f"  {feat:<28} {coef:>9.4f}  {scaled_ml:>12.1f}  {v1_val:>13}  {direction}")

    print(f"\n  Intercept: {intercept:.4f}")
    print(f"\n  Interpretation:")
    print(f"    Positive coef = feature pushes toward unsafe (1)")
    print(f"    Negative coef = feature pushes toward safe (0)")
    print(f"    'ML→85 scale' normalises ML coefficients to the same 0–85 range")
    print(f"    as v1 deductions so you can compare them directly.")


# ── 6. Save ───────────────────────────────────────────────────────────────────

def save_model(pipe: Pipeline):
    print(banner("6. SAVE MODEL"))
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"  Saved → {MODEL_PATH}  ({size_kb:.1f} KB)")
    print(f"\n  To load later:")
    print(f"    import joblib")
    print(f"    model = joblib.load('{MODEL_PATH}')")
    print(f"    prob  = model.predict_proba(X_new)[:, 1]")
    print(f"    pred  = (prob >= {DECISION_THRESHOLD}).astype(int)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="allergen_dataset_v6_scored.csv",
                        help="Path to the scored CSV file")
    args = parser.parse_args()

    print(f"\n{'═'*W}")
    print(f"  PuroScore v1 Model Trainer".center(W))
    print(f"  trainPuroScoreModel.py".center(W))
    print(f"{'═'*W}")

    X, y, df_full = load_and_prep(args.csv)
    X_train, X_test, y_train, y_test = split(X, y)
    pipe = train(X_train, y_train)
    evaluate(pipe, X_test, y_test, df_full, X)
    print_coefficients(pipe)
    save_model(pipe)

    print(f"\n{'═'*W}")
    print(f"  Done.".center(W))
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
