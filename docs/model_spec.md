# PuroScore v2 — Model Specification

## v1 Rules Engine (Deterministic Baseline)

### Deduction Table (src/rules_engine.py)

| Trigger                        | Deduction | Notes                            |
|-------------------------------|-----------|----------------------------------|
| Direct allergen ingredient     | 85        | Keyword found in ingredient list |
| Contains statement             | 85        | "contains:" / "allergens:" text  |
| Derived ingredient             | 75        | Synonym / alias resolution       |
| May contain statement          | 45        | Precautionary cross-contact      |
| Shared facility / equipment    | 35        | Manufacturing disclosure         |
| Ambiguous term (each)          | 10        | "natural flavors", "spices", etc |
| Missing / unreadable data      | 50        | Blank ingredient field           |

### Confidence Modifiers

| Confidence | Modifier | Condition                     |
|------------|----------|-------------------------------|
| High       | 0        | Low ambiguity, good data      |
| Medium     | 8        | 3+ ambiguous terms            |
| Low        | 18        | Missing / incomplete data     |

### Formula

```
score = clamp(0..100,  100 − Σ(deductions) − (ambiguity_count × 5) − confidence_modifier)
```

---

## v2 ML Layer (Logistic Regression)

### Training Setup

| Item                  | Value                                    |
|-----------------------|------------------------------------------|
| Model                 | Logistic Regression                      |
| Solver                | saga (handles large feature spaces)      |
| Max iterations        | 3,000                                    |
| Class weight          | balanced (handles allergen imbalance)    |
| Scaling               | StandardScaler                           |
| Calibration           | Platt scaling (CalibratedClassifierCV)   |
| Decision threshold    | 0.30 (recall-optimised)                  |
| Training rows         | 1,000 (allergen_dataset_v6)              |
| Training unit         | 1 row per (product × allergen)           |

### Training Target

```python
unsafe = Contains_[allergen] | MayContain_[allergen]
# unsafe = 1 → product has or may have this allergen
# unsafe = 0 → no evidence of this allergen
```

### Feature Set

| Feature                  | Source                        | Type     |
|--------------------------|-------------------------------|----------|
| kw_[allergen] × 9        | Keyword scan of text          | Binary   |
| has_contains_stmt        | "contains:" / "allergens:"    | Binary   |
| has_may_contain          | "may contain" text            | Binary   |
| has_shared_facility      | Facility/equipment text       | Binary   |
| ambiguity_count          | Count of ambiguous terms      | Integer  |
| ingredient_count         | Number of ingredients         | Integer  |
| CrossContamination_Flag  | Dataset pre-label / text      | Binary   |
| Spices_Flag              | "spices" in text              | Binary   |
| NaturalFlavors_Flag      | "natural flavors" in text     | Binary   |
| v1_score                 | Rules engine output           | Float    |
| v1_conf_high/medium/low  | Confidence tier               | Binary   |
| trigger_count            | Number of keyword matches     | Integer  |
| rule_deduction           | Total v1 deduction applied    | Float    |
| has_may_contain_txt      | May-contain text detected     | Binary   |
| has_cross_cont_txt       | Cross-contamination detected  | Binary   |
| product_category         | LabelEncoded Food_Name        | Integer  |
| TF-IDF (200 features)    | Bigram TF-IDF on ingredients  | Float    |

### Evaluation Results (5-fold CV on v6 dataset)

| Allergen             | Recall | Precision | Contains | MayContain | Unsafe | Total |
|----------------------|--------|-----------|----------|------------|--------|-------|
| Milk (Dairy)         | 0.899  | 0.948     | 358      | 46         | 404    | 1000  |
| Eggs                 | 0.827  | 0.742     | 62       | 13         | 75     | 1000  |
| Fish                 | 0.910  | 0.920     | 17       | 4          | 21     | 1000  |
| Crustacean Shellfish | 0.400  | 0.367     | 3        | 4          | 7      | 1000  |
| Tree Nuts            | 0.819  | 0.791     | 242      | 45         | 287    | 1000  |
| Peanuts              | 0.863  | 0.827     | 97       | 20         | 117    | 1000  |
| Wheat                | 0.901  | 0.914     | 301      | 115        | 416    | 1000  |
| Soybeans             | 0.901  | 0.914     | 267      | 67         | 334    | 1000  |
| Sesame               | 0.805  | 0.682     | 57       | 15         | 72     | 1000  |

> **Note on Crustacean:** Low recall (0.40) is expected — only 7 unsafe samples out of 1,000.
> More data will fix this naturally. Do not increase threshold to compensate.

---

## trainPuroScoreModel.py (Feature-Based Trainer)

Separate script for training on the structured `allergen_dataset_v6_scored.csv`
(9,000 rows, one per product × allergen, with pre-engineered feature columns).

### Input CSV Schema

| Column                  | Type    | Description                          |
|-------------------------|---------|--------------------------------------|
| product_name            | string  | Product identifier                   |
| allergen                | string  | Allergen name                        |
| has_direct_allergen     | binary  | Direct ingredient keyword match      |
| has_derived_allergen    | binary  | Synonym / derived mapping            |
| has_contains_statement  | binary  | "contains:" declaration              |
| has_may_contain         | binary  | Precautionary statement              |
| has_shared_facility     | binary  | Facility / equipment disclosure      |
| ambiguity_count         | integer | Count of ambiguous terms             |
| missing_data_flag       | binary  | Ingredient field blank/unreadable    |
| ocr_confidence          | float   | OCR quality score (0.0–1.0)          |
| puro_score              | float   | v1 rule engine score (0–100)         |
| confidence              | string  | High / Medium / Low                  |
| triggers                | string  | Comma-separated matched keywords     |
| label                   | binary  | unsafe=1, safe/uncertain=0           |

### Output

- Trained model saved as `models/puroScoreModel_v1.joblib`
- Printed evaluation: recall, precision, FNR (overall + per allergen)
- Coefficient table vs v1 hardcoded deductions (normalised to same scale)
