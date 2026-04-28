"""
PuroScore v2 — ML Model Layer
Updated for allergen_dataset_v6 (Contains + MayContain separate columns).

Training target per allergen:
  unsafe = 1  if  Contains_[a] == 1  OR  MayContain_[a] == 1
  unsafe = 0  otherwise

This lets the model learn the full risk signal. At score time, the
blend weights the v1 rule score (which already distinguishes contains
vs. may-contain via deduction levels) against the ML probability.

Per spec Section 11.1: Logistic Regression, calibrated, recall-optimised.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.feature_engineering import PuroFeaturizer, ALLERGEN_COL

ALLERGENS = list(ALLERGEN_COL.keys())

# Blend weights — per spec Option B: Final Risk = α×Rule + (1-α)×ML
ALPHA = 0.35   # rules weight
BETA  = 0.65   # ML weight

# Recall-optimised decision threshold (lower = higher recall, fewer false negatives)
RECALL_THRESHOLD = 0.30


class PuroV2Model:
    """
    PuroScore v2 — Hybrid (Rules + Logistic Regression) multi-allergen engine.

    Two-layer architecture per spec:
      Layer 1 (this class)  → product-level scores + confidence
      Layer 2 (PuroStatus)  → user severity → Red/Yellow/Green  [NOT implemented here]
    """

    def __init__(self):
        self.models: dict       = {}
        self.featurizer         = PuroFeaturizer(max_tfidf_features=200)
        self.eval_results: dict = {}
        self._trained           = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, verbose: bool = True) -> "PuroV2Model":
        if verbose:
            print(f"\n  Training PuroScore v2 — {len(df)} samples, {len(ALLERGENS)} allergens")
            print(f"  Dataset columns: {[c for c in df.columns if 'Contains' in c or 'Flag' in c]}")
            print(f"  Blend: α={ALPHA} (rules) + β={BETA} (ML)\n")

        self.featurizer.fit(df)

        for allergen in ALLERGENS:
            contains_col, may_contain_col = ALLERGEN_COL[allergen]

            # Build combined unsafe label: direct OR precautionary presence
            contains    = df[contains_col].fillna(0).astype(int)   if contains_col    in df.columns else pd.Series(0, index=df.index)
            may_contain = df[may_contain_col].fillna(0).astype(int) if may_contain_col in df.columns else pd.Series(0, index=df.index)
            y = (contains | may_contain).values

            X = self.featurizer.transform(df, allergen)
            classes = np.unique(y)

            if len(classes) < 2:
                if verbose:
                    print(f"  [SKIP] {allergen:<14} — only one class in labels")
                continue

            weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = dict(zip(classes, weights))

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="saga",
                    class_weight=class_weight,
                    random_state=42,
                )),
            ])
            calibrated = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
            calibrated.fit(X, y)
            self.models[allergen] = calibrated

            # Cross-val metrics
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            recall_cv = cross_val_score(pipe, X, y, cv=cv, scoring="recall")
            prec_cv   = cross_val_score(pipe, X, y, cv=cv, scoring="precision")

            self.eval_results[allergen] = {
                "recall_mean":    round(float(np.mean(recall_cv)), 3),
                "recall_std":     round(float(np.std(recall_cv)), 3),
                "precision_mean": round(float(np.mean(prec_cv)), 3),
                "n_contains":     int(contains.sum()),
                "n_may_contain":  int(may_contain.sum()),
                "n_unsafe":       int(y.sum()),
                "n_total":        int(len(y)),
            }

            if verbose:
                r = self.eval_results[allergen]
                print(
                    f"  {allergen:<14} Recall {r['recall_mean']:.3f} ±{r['recall_std']:.3f}"
                    f"  Prec {r['precision_mean']:.3f}"
                    f"  | Contains {r['n_contains']:>3}  MayCont {r['n_may_contain']:>3}"
                    f"  → unsafe {r['n_unsafe']:>3}/{r['n_total']}"
                )

        self._trained = True
        if verbose:
            print(f"\n  ✓ Training complete — {len(self.models)} models ready\n")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_product(self, product_name: str, label_description: str) -> dict:
        """
        Score a single product across all 9 allergens.
        Returns per-allergen dict with v1_score, ml_prob, final_score,
        confidence, triggers, has_may_contain.
        """
        assert self._trained, "Call train() before predict_product()."

        from src.rules_engine import score_one

        row_df = pd.DataFrame([{
            "Food_Name":         product_name,
            "Label_Description": label_description,
        }])

        results = {}
        for allergen in ALLERGENS:
            # Layer A — v1 rule score
            rule_out = score_one(label_description, allergen)
            v1_score = rule_out["score"]

            # Layer B — ML probability
            if allergen in self.models:
                X = self.featurizer.transform(row_df, allergen)
                ml_prob = float(self.models[allergen].predict_proba(X)[0][1])
            else:
                ml_prob = 1.0 - (v1_score / 100.0)

            ml_score = 100.0 * (1.0 - ml_prob)

            # Layer C — v2 hybrid blend
            final_score = ALPHA * v1_score + BETA * ml_score

            # Confidence from calibrated probability distance from 0.5
            certainty = max(ml_prob, 1.0 - ml_prob)
            if rule_out["confidence"] == "Low":
                confidence = "Low"
            elif certainty >= 0.80:
                confidence = "High"
            elif certainty >= 0.60:
                confidence = "Medium"
            else:
                confidence = "Low"

            results[allergen] = {
                "final_score":     round(final_score, 1),
                "v1_score":        v1_score,
                "ml_prob":         round(ml_prob, 4),
                "ml_score":        round(ml_score, 1),
                "confidence":      confidence,
                "triggers":        rule_out["triggers"],
                "ambiguity":       rule_out["ambiguity_count"],
                "has_may_contain": rule_out["has_may_contain"],
                "unsafe":          ml_prob >= RECALL_THRESHOLD,
            }

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "models/puro_v2.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/puro_v2.pkl") -> "PuroV2Model":
        with open(path, "rb") as f:
            return pickle.load(f)
