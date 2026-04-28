# PuroScore v2 — Quickstart

## Setup

```bash
git clone https://github.com/NathanielKung/PuroPlate_ML.git
cd PuroPlate_ML
pip install -r requirements.txt
```

Place `allergen_dataset_v6.xlsx` in the project root.

---

## Run the consultation engine (v2 hybrid)

```bash
# Interactive — type any product + ingredients
python3 main.py

# 3 built-in demo products
python3 main.py --demo

# Force retrain (rebuilds models/puro_v2.pkl from scratch)
python3 main.py --retrain
```

---

## Train the feature-based v1 model

```bash
# Requires allergen_dataset_v6_scored.csv in the project root
python3 trainPuroScoreModel.py

# Custom CSV path
python3 trainPuroScoreModel.py --csv path/to/your_scored_data.csv
```

Outputs `models/puroScoreModel_v1.joblib`.

---

## Use a saved model in your own code

```python
import joblib
import pandas as pd

# Load
model = joblib.load("models/puroScoreModel_v1.joblib")

# Single product row
features = pd.DataFrame([{
    "has_direct_allergen":    1,
    "has_derived_allergen":   0,
    "has_contains_statement": 1,
    "has_may_contain":        0,
    "has_shared_facility":    0,
    "ambiguity_count":        2,
    "missing_data_flag":      0,
    "ocr_confidence":         0.95,
}])

prob  = model.predict_proba(features)[:, 1][0]   # P(unsafe)
score = 100 * (1 - prob)                          # PuroScore (0–100)
unsafe = prob >= 0.30                             # recall-optimised threshold

print(f"PuroScore: {score:.1f}  |  P(unsafe): {prob:.2%}  |  Flagged: {unsafe}")
```

---

## File Structure

```
PuroPlate_ML/
├── main.py                        ← v2 hybrid engine entry point
├── trainPuroScoreModel.py         ← feature-based LR trainer
├── requirements.txt
├── allergen_dataset_v6.xlsx       ← 1,000-row training dataset
├── allergen_dataset_v6_scored.csv ← 9,000-row scored dataset (add this)
├── models/
│   ├── puro_v2.pkl                ← v2 hybrid model (auto-generated)
│   └── puroScoreModel_v1.joblib   ← v1 feature model (from trainPuroScoreModel.py)
├── src/
│   ├── rules_engine.py            ← v1 deterministic scorer
│   ├── feature_engineering.py     ← feature extraction pipeline
│   ├── ml_model.py                ← v2 hybrid model
│   └── consult.py                 ← consultation report renderer
└── docs/
    ├── architecture.md            ← system architecture
    ├── model_spec.md              ← full model specification
    └── quickstart.md              ← this file
```

---

## Two things NOT in this repo (by design)

1. **PuroStatus (user layer)** — takes PuroScore outputs + user severity profile → Red/Yellow/Green. Belongs in the app layer, not here.
2. **v3 ML-driven engine** — out of scope for this release.
