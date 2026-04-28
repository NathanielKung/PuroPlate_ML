# PuroScore v2 — Architecture Overview

## Two-Layer System (per spec)

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — PuroScore (Product-Level)                │
│  src/rules_engine.py + src/ml_model.py              │
│                                                     │
│  Input : ingredient text                            │
│  Output: score (0–100), confidence, triggers        │
│  Rule  : NEVER returns Red/Yellow/Green             │
└──────────────────────┬──────────────────────────────┘
                       │  scores + confidence
┌──────────────────────▼──────────────────────────────┐
│  LAYER 2 — PuroStatus (User-Level)  [NOT IN REPO]   │
│                                                     │
│  Input : Layer 1 scores + user severity profile     │
│  Output: Red / Yellow / Green + UI behavior         │
└─────────────────────────────────────────────────────┘
```

---

## v2 Hybrid Score Formula

```
Final Score = α × RuleScore + β × MLScore

where:
  RuleScore = v1 deterministic output (0–100)
  MLScore   = 100 × (1 − P(unsafe))
  α = 0.35  (rules weight)
  β = 0.65  (ML weight)
```

---

## End-to-End Flow

```
Raw ingredient text
        ↓
┌───────────────────────┐
│  rules_engine.py      │  Keyword matching → deduction table → v1 score
│  Layer A              │  Signals: direct / contains / may_contain / facility
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  feature_engineering  │  TF-IDF (200 features, bigrams)
│  .py  Layer B         │  + keyword flags per allergen
│                       │  + rule engine outputs (score, deduction, triggers)
│                       │  + CrossContamination / Spices / NaturalFlavors flags
│                       │  + product category encoding
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  ml_model.py          │  9 × Logistic Regression (one per allergen)
│  Layer C              │  Pipeline: StandardScaler → LR (saga, balanced)
│                       │  Calibrated with Platt scaling
│                       │  Threshold: 0.30 (recall-optimised)
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  consult.py           │  Renders professional consultation report
│  Output               │  Score / ML risk % / Signal type / Triggers
└───────────────────────┘
```

---

## Score Interpretation

| PuroScore | Meaning                        |
|-----------|-------------------------------|
| 0–25      | High risk — allergen confirmed |
| 25–50     | Caution — strong signals       |
| 50–75     | Low risk — weak signals        |
| 75–100    | Safe — no signals detected     |

---

## Allergens Covered (9)

| Internal Key  | Display Name         | Dataset Columns                             |
|---------------|----------------------|---------------------------------------------|
| Milk          | Milk (Dairy)         | Contains_Milk / MayContain_Milk             |
| Eggs          | Eggs                 | Contains_Eggs / MayContain_Eggs             |
| Fish          | Fish                 | Contains_Fish / MayContain_Fish             |
| Crustacean    | Crustacean Shellfish | Contains_Crustacean / MayContain_Crustacean |
| TreeNuts      | Tree Nuts            | Contains_TreeNuts / MayContain_TreeNuts     |
| Peanuts       | Peanuts              | Contains_Peanuts / MayContain_Peanuts       |
| Wheat         | Wheat                | Contains_Wheat / MayContain_Wheat           |
| Soybeans      | Soybeans             | Contains_Soybeans / MayContain_Soybeans     |
| Sesame        | Sesame               | Contains_Sesame / MayContain_Sesame         |
