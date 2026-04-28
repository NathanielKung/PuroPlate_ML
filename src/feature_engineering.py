"""
PuroScore v2 — Feature Engineering Layer
Converts raw ingredient text into structured feature vectors for ML.
Per spec Section 9: "the model does not read 'ingredients' magically."
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.rules_engine import ALLERGEN_KEYWORDS, AMBIGUOUS_TERMS, score_one

# ── Keyword feature sets ─────────────────────────────────────────────────────

CONTAINS_PATTERNS   = [r"contains\s*:", r"\bcontains\b"]
MAY_CONTAIN_PATTERNS = [r"may contain", r"might contain", r"could contain"]
FACILITY_PATTERNS   = [r"produced in a facility", r"shared equipment", r"manufactured in",
                        r"processed in a facility", r"made on equipment"]
FREE_FROM_PATTERNS  = [r"free\b", r"certified.*free", r"dedicated.*free", r"no\s+\w+\s+ingredient"]


def _flag(text: str, patterns: list) -> int:
    text = text.lower()
    return int(any(re.search(p, text) for p in patterns))


def extract_keyword_features(ingredients: str) -> dict:
    """Binary presence flags for each allergen + statement flags."""
    text = str(ingredients).lower()
    feats = {}

    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        feats[f"kw_{allergen}"] = int(any(kw in text for kw in keywords))

    feats["has_contains_stmt"]    = _flag(text, CONTAINS_PATTERNS)
    feats["has_may_contain"]      = _flag(text, MAY_CONTAIN_PATTERNS)
    feats["has_shared_facility"]  = _flag(text, FACILITY_PATTERNS)
    feats["has_free_from"]        = _flag(text, FREE_FROM_PATTERNS)
    feats["ambiguity_count"]      = sum(1 for t in AMBIGUOUS_TERMS if t in text)
    feats["ingredient_count"]     = len([i.strip() for i in text.split(",") if i.strip()])

    return feats


def extract_rule_features(ingredients: str, allergen: str) -> dict:
    """Rule engine outputs for a single allergen — used as ML input features."""
    result = score_one(ingredients, allergen)
    return {
        "v1_score":         result["score"],
        "v1_conf_high":     int(result["confidence"] == "High"),
        "v1_conf_medium":   int(result["confidence"] == "Medium"),
        "v1_conf_low":      int(result["confidence"] == "Low"),
        "trigger_count":    len(result["triggers"]),
        "ambiguity_count":  result["ambiguity_count"],
        "rule_deduction":   result["rule_deduction"],
    }


class PuroFeaturizer:
    """
    Transforms a DataFrame of products into the full feature matrix.
    Fit on training data; transform on inference.
    """

    def __init__(self, max_tfidf_features: int = 150):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z ]+\b",
        )
        self.label_enc = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self.tfidf.fit(df["Ingredients"].fillna("").astype(str))
        self.label_enc.fit(df["Label"].fillna("Unknown").astype(str))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, allergen: str) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()."

        rows = []
        for _, row in df.iterrows():
            ingredients = str(row.get("Ingredients", "") or "")
            label       = str(row.get("Label", "Unknown") or "Unknown")

            kw_feats   = extract_keyword_features(ingredients)
            rule_feats = extract_rule_features(ingredients, allergen)

            # Category encoding
            try:
                cat = self.label_enc.transform([label])[0]
            except ValueError:
                cat = -1

            structured = list(kw_feats.values()) + list(rule_feats.values()) + [cat]
            rows.append(structured)

        structured_arr = np.array(rows, dtype=float)

        # TF-IDF
        tfidf_arr = self.tfidf.transform(
            df["Ingredients"].fillna("").astype(str)
        ).toarray()

        return np.hstack([structured_arr, tfidf_arr])

    def fit_transform(self, df: pd.DataFrame, allergen: str) -> np.ndarray:
        self.fit(df)
        return self.transform(df, allergen)
