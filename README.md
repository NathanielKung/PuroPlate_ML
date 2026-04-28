# PuroPlate ML — PuroScore v2

**Hybrid allergen risk scoring engine: Rules + Logistic Regression**

Built to spec from *PuroScore ML Model Approach* — implements **Phase 2 (Hybrid)** only. v3 (full ML-driven) is out of scope for this release.

---

## Architecture

```
Raw Ingredients Text
        ↓
┌─────────────────────────────────┐
│   Layer A — v1 Rules Engine     │  Deterministic baseline score (0–100)
│   (rules_engine.py)             │  Hard truth: direct/derived/may-contain
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Layer B — Feature Engineering │  Keyword flags, TF-IDF, rule outputs,
│   (feature_engineering.py)      │  ambiguity count, category encoding
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Layer C — ML Signal           │  9 calibrated Logistic Regression models
│   (ml_model.py)                 │  One per allergen → P(unsafe)
└─────────────────────────────────┘
        ↓
   Final Score = α×RuleScore + β×MLScore
   α = 0.35  (rules)   β = 0.65  (ML)
        ↓
┌─────────────────────────────────┐
│   Consultation Report           │  Professional per-allergen output
│   (consult.py)                  │  Score, confidence, triggers, ML risk %
└─────────────────────────────────┘
```

**Layer separation is strict:**
- `PuroScore` (this engine) = product-level only. No UI colors.
- `PuroStatus` (not in this repo) = user-severity layer that outputs Red/Yellow/Green.

---

## Allergens Covered

Milk · Egg · Peanut · Tree Nuts · Soy · Wheat · Fish · Crustacean Shellfish · Sesame

---

## Setup

```bash
pip install -r requirements.txt
```

Place `test_dataset.xlsx` in the project root (500-row labeled dataset with binary allergen columns).

---

## Usage

```bash
# Interactive consultation mode
python main.py

# 3 built-in demo products
python main.py --demo

# Train + save model only
python main.py --train-only

# Force retrain (ignore saved model)
python main.py --retrain
```

---

## Example Output

```
════════════════════════════════════════════════════════════════════
              PUROPLATE ALLERGEN CONSULTATION REPORT
              PuroScore v2  ·  Hybrid Rules + ML Engine
════════════════════════════════════════════════════════════════════
  Product     : Caesar Dressing
  Ingredients : SOYBEAN OIL, WATER, PARMESAN CHEESE, ANCHOVIES, EGG YOLK...
  Analyzed    : 2026-04-28
────────────────────────────────────────────────────────────────────
  ALLERGEN RISK PROFILE
────────────────────────────────────────────────────────────────────
  Allergen               Score  ML Risk    Conf  Bar (safe →)  Status
  ──────────────────────────────────────────────────────────────────
  Milk (Dairy)           12.4     89%    High  ████░░░░░░░░░░░░░░░░  ⚠ HIGH RISK
  Egg                    14.1     87%    High  ██░░░░░░░░░░░░░░░░░░  ⚠ HIGH RISK
  Fish                   13.5     88%    High  ███░░░░░░░░░░░░░░░░░  ⚠ HIGH RISK
  Soy                    15.0     85%    High  ███░░░░░░░░░░░░░░░░░  ⚠ HIGH RISK
  ...
```

---

## Model Details

| Item | Spec |
|---|---|
| Model | Logistic Regression (calibrated, per allergen) |
| Training unit | 1 row per (product × allergen) |
| Target | unsafe=1, safe=0 |
| Evaluation metric | Recall (false-negatives are highest-risk failure) |
| Blend | α=0.35 rules + β=0.65 ML |
| Threshold | 0.35 (recall-optimised) |
| Calibration | Platt scaling via `CalibratedClassifierCV` |

---

## File Structure

```
PuroPlate_ML/
├── main.py                   ← entry point
├── requirements.txt
├── test_dataset.xlsx         ← 500-row training data
├── models/
│   └── puro_v2.pkl           ← saved model (auto-generated on first run)
└── src/
    ├── rules_engine.py       ← v1 deterministic scorer
    ├── feature_engineering.py← feature extraction pipeline
    ├── ml_model.py           ← v2 hybrid model
    └── consult.py            ← consultation report renderer
```
