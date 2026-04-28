"""
PuroScore v2 — Feature Engineering Layer
Updated for allergen_dataset_v6 schema.

New signals from the richer dataset:
  - CrossContamination_Flag  → pre-labeled cross-contact risk
  - Spices_Flag              → ambiguous "spices" present
  - NaturalFlavors_Flag      → ambiguous "natural flavors" present
  - Separate Contains / MayContain columns per allergen

At inference time these flags are recomputed from ingredient text
(they're not available as pre-labeled columns for new products).
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.rules_engine import ALLERGEN_KEYWORDS, AMBIGUOUS_TERMS, score_one, CROSS_CONTAMINATION_PHRASES

# ── Allergen column mapping (v6 dataset names) ───────────────────────────────
ALLERGEN_COL = {
    "Milk":       ("Contains_Milk",       "MayContain_Milk"),
    "Eggs":       ("Contains_Eggs",       "MayContain_Eggs"),
    "Fish":       ("Contains_Fish",       "MayContain_Fish"),
    "Crustacean": ("Contains_Crustacean", "MayContain_Crustacean"),
    "TreeNuts":   ("Contains_TreeNuts",   "MayContain_TreeNuts"),
    "Peanuts":    ("Contains_Peanuts",    "MayContain_Peanuts"),
    "Wheat":      ("Contains_Wheat",      "MayContain_Wheat"),
    "Soybeans":   ("Contains_Soybeans",   "MayContain_Soybeans"),
    "Sesame":     ("Contains_Sesame",     "MayContain_Sesame"),
}

CONTAINS_PATTERNS    = [r"contains\s*:", r"\bcontains\b", r"allergens\s*:"]
MAY_CONTAIN_PATTERNS = [r"may contain", r"might contain", r"could contain", r"peut contenir"]
FACILITY_PATTERNS    = [r"produced in a facility", r"shared equipment", r"manufactured in",
                         r"processed in a facility", r"made on equipment", r"traces of"]


def _flag(text: str, patterns: list) -> int:
    return int(any(re.search(p, text, re.IGNORECASE) for p in patterns))


def _compute_flags(text: str) -> dict:
    """
    Recompute dataset FLAGS from raw text — used at inference for new products.
    Mirrors CrossContamination_Flag, Spices_Flag, NaturalFlavors_Flag.
    """
    t = text.lower()
    return {
        "CrossContamination_Flag": int(any(p in t for p in CROSS_CONTAMINATION_PHRASES)),
        "Spices_Flag":             int("spices" in t or "seasoning" in t or "épices" in t),
        "NaturalFlavors_Flag":     int("natural flavor" in t or "natural flavour" in t
                                       or "arôme" in t or "arome" in t),
    }


def extract_keyword_features(ingredients: str) -> dict:
    """Binary presence flags for each allergen keyword + statement flags."""
    text = str(ingredients).lower()
    feats = {}
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        feats[f"kw_{allergen}"] = int(any(kw in text for kw in keywords))

    feats["has_contains_stmt"]   = _flag(text, CONTAINS_PATTERNS)
    feats["has_may_contain"]     = _flag(text, MAY_CONTAIN_PATTERNS)
    feats["has_shared_facility"] = _flag(text, FACILITY_PATTERNS)
    feats["ambiguity_count"]     = sum(1 for t in AMBIGUOUS_TERMS if t in text)
    feats["ingredient_count"]    = len([i.strip() for i in text.split(",") if i.strip()])

    # FLAGS — computed from text, match the pre-labeled dataset columns
    flags = _compute_flags(text)
    feats.update(flags)

    return feats


def extract_rule_features(ingredients: str, allergen: str) -> dict:
    """v1 rule engine outputs for a single allergen — used as ML input features."""
    r = score_one(ingredients, allergen)
    return {
        "v1_score":            r["score"],
        "v1_conf_high":        int(r["confidence"] == "High"),
        "v1_conf_medium":      int(r["confidence"] == "Medium"),
        "v1_conf_low":         int(r["confidence"] == "Low"),
        "trigger_count":       len(r["triggers"]),
        "ambiguity_count":     r["ambiguity_count"],
        "rule_deduction":      r["rule_deduction"],
        "has_may_contain_txt": int(r["has_may_contain"]),
        "has_cross_cont_txt":  int(r["has_cross_contamination"]),
    }


class PuroFeaturizer:
    """
    Transforms a DataFrame into the full feature matrix.
    Handles both training data (with pre-labeled FLAG columns) and
    inference data (where FLAGs are recomputed from text).
    """

    def __init__(self, max_tfidf_features: int = 200):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z ]+\b",
            sublinear_tf=True,
        )
        self.label_enc = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        texts = df["Label_Description"].fillna("").astype(str)
        self.tfidf.fit(texts)
        self.label_enc.fit(df["Food_Name"].fillna("Unknown").astype(str))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, allergen: str) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()."

        rows = []
        for _, row in df.iterrows():
            ingr  = str(row.get("Label_Description", "") or "")
            name  = str(row.get("Food_Name", "Unknown") or "Unknown")

            kw    = extract_keyword_features(ingr)
            rule  = extract_rule_features(ingr, allergen)

            # Dataset pre-labeled FLAGS (if available) override text-computed ones
            if "CrossContamination_Flag" in row.index:
                kw["CrossContamination_Flag"] = int(row.get("CrossContamination_Flag", 0) or 0)
                kw["Spices_Flag"]             = int(row.get("Spices_Flag", 0) or 0)
                kw["NaturalFlavors_Flag"]     = int(row.get("NaturalFlavors_Flag", 0) or 0)

            # Category
            try:
                cat = self.label_enc.transform([name])[0]
            except ValueError:
                cat = -1

            rows.append(list(kw.values()) + list(rule.values()) + [cat])

        structured = np.array(rows, dtype=float)
        tfidf_arr  = self.tfidf.transform(
            df["Label_Description"].fillna("").astype(str)
        ).toarray()

        return np.hstack([structured, tfidf_arr])

    def fit_transform(self, df: pd.DataFrame, allergen: str) -> np.ndarray:
        self.fit(df)
        return self.transform(df, allergen)
