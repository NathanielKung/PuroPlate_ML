"""
PuroScore v2 — ML Model Layer
Binary Relevance: one calibrated Logistic Regression per allergen.
Per spec Section 11.1: "Learn better weights for deterministic features."
Priority metric: Recall (avoid false negatives — per spec Section 16).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.feature_engineering import PuroFeaturizer

ALLERGENS = ["Milk", "Egg", "Peanut", "Tree_Nuts", "Soy", "Wheat", "Fish", "Shellfish", "Sesame"]

# Blend weights — per spec Option B: Final Risk = α×Rule + (1-α)×ML
# v2 gives ML more weight than rules but rules remain the floor
ALPHA = 0.35   # rules weight
BETA  = 0.65   # ML weight

# Recall-optimised decision threshold (lower = higher recall, fewer false negatives)
RECALL_THRESHOLD = 0.35


class PuroV2Model:
    """
    PuroScore v2 — Hybrid (Rules + Logistic Regression) multi-allergen engine.
    Scores one product × allergen pair as: Final Score = α×RuleScore + β×MLScore
    where MLScore = 100 × (1 - P(unsafe)).
    """

    def __init__(self):
        self.models: dict[str, CalibratedClassifierCV] = {}
        self.featurizer = PuroFeaturizer(max_tfidf_features=150)
        self.eval_results: dict[str, dict] = {}
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, verbose: bool = True) -> "PuroV2Model":
        """
        Train one calibrated LR per allergen on the dataset.
        Uses class-weight balancing + recall-optimised threshold.
        """
        if verbose:
            print(f"\n  Training PuroScore v2 — {len(df)} samples, {len(ALLERGENS)} allergens")
            print(f"  Blend: α={ALPHA} (rules) + β={BETA} (ML)\n")

        # Fit featurizer on full dataset once
        self.featurizer.fit(df)

        for allergen in ALLERGENS:
            if allergen not in df.columns:
                if verbose:
                    print(f"  [SKIP] {allergen} — column not found")
                continue

            y = df[allergen].fillna(0).astype(int).values
            X = self.featurizer.transform(df, allergen)

            # Class weights — handles imbalance (some allergens are rare)
            classes = np.unique(y)
            if len(classes) < 2:
                if verbose:
                    print(f"  [SKIP] {allergen} — only one class in labels")
                continue

            weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = dict(zip(classes, weights))

            base_lr = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight=class_weight,
                random_state=42,
            )

            # Calibrate for reliable probabilities (Platt scaling)
            calibrated = CalibratedClassifierCV(base_lr, cv=3, method="sigmoid")
            calibrated.fit(X, y)
            self.models[allergen] = calibrated

            # Cross-val evaluation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            recall_scores = cross_val_score(base_lr, X, y, cv=cv, scoring="recall")
            prec_scores   = cross_val_score(base_lr, X, y, cv=cv, scoring="precision")

            self.eval_results[allergen] = {
                "recall_mean":    round(float(np.mean(recall_scores)), 3),
                "recall_std":     round(float(np.std(recall_scores)), 3),
                "precision_mean": round(float(np.mean(prec_scores)), 3),
                "positive_rate":  round(float(y.mean()), 3),
                "n_positive":     int(y.sum()),
                "n_total":        int(len(y)),
            }

            if verbose:
                r = self.eval_results[allergen]
                print(f"  {allergen:<14} Recall {r['recall_mean']:.3f} ±{r['recall_std']:.3f} "
                      f"| Prec {r['precision_mean']:.3f} "
                      f"| Positives {r['n_positive']}/{r['n_total']}")

        self._trained = True
        if verbose:
            print(f"\n  ✓ Training complete — {len(self.models)} models ready\n")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_product(self, product_name: str, ingredients: str) -> dict:
        """
        Score a single product across all 9 allergens.
        Returns per-allergen dict with v1_score, ml_prob, final_score, confidence, triggers.
        """
        assert self._trained, "Call train() before predict_product()."

        from src.rules_engine import score_one, ALLERGEN_KEYWORDS

        # Build a 1-row DataFrame for the featurizer
        row_df = pd.DataFrame([{"Label": product_name, "Ingredients": ingredients}])

        results = {}
        for allergen in ALLERGENS:
            # ── Layer A: v1 rule score ──────────────────────────────────────
            rule_out  = score_one(ingredients, allergen)
            v1_score  = rule_out["score"]           # 0–100, lower = riskier

            # ── Layer B: ML probability ─────────────────────────────────────
            if allergen in self.models:
                X       = self.featurizer.transform(row_df, allergen)
                ml_prob = float(self.models[allergen].predict_proba(X)[0][1])  # P(unsafe)
            else:
                ml_prob = 1.0 - (v1_score / 100.0)  # Fall back to rule-derived prob

            ml_score  = 100 * (1.0 - ml_prob)       # 0–100, higher = safer

            # ── Layer C: v2 hybrid blend ────────────────────────────────────
            final_score = ALPHA * v1_score + BETA * ml_score

            # ── Confidence from ml_prob calibration ─────────────────────────
            certainty = max(ml_prob, 1.0 - ml_prob)  # distance from 0.5
            if certainty >= 0.80:
                confidence = "High"
            elif certainty >= 0.60:
                confidence = "Medium"
            else:
                confidence = "Low"

            # Override: Low confidence if rule engine already flagged it
            if rule_out["confidence"] == "Low":
                confidence = "Low"

            # Unsafe flag uses recall-optimised threshold
            unsafe = ml_prob >= RECALL_THRESHOLD

            results[allergen] = {
                "final_score":  round(final_score, 1),
                "v1_score":     v1_score,
                "ml_prob":      round(ml_prob, 4),
                "ml_score":     round(ml_score, 1),
                "confidence":   confidence,
                "triggers":     rule_out["triggers"],
                "ambiguity":    rule_out["ambiguity_count"],
                "unsafe":       unsafe,
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
