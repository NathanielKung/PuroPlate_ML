"""
PuroScore v1 — Deterministic Rules Engine
Layer 1 (Product-Level) baseline scorer.
Per spec: rules = hard truth. No UI colors returned here.

Updated for allergen_dataset_v6 column schema:
  Contains_[allergen]    → direct presence (deduction 85)
  MayContain_[allergen]  → precautionary (deduction 45)
  CrossContamination_Flag / Spices_Flag / NaturalFlavors_Flag → ambiguity
"""

ALLERGEN_KEYWORDS = {
    "Milk":       ["milk", "dairy", "whey", "casein", "caseinate", "lactose", "butter",
                   "cream", "cheese", "milk powder", "milk solids", "parmesan", "cheddar",
                   "mozzarella", "buttermilk", "ghee", "yogurt", "skim milk", "whole milk",
                   "nonfat milk", "sodium caseinate", "lactalbumin", "lactoferrin",
                   "parmigiano", "grana padano", "ricotta", "brie", "beurre"],
    "Eggs":       ["egg", "eggs", "egg whites", "egg white", "egg yolk", "egg albumen",
                   "whole egg", "mayonnaise", "albumin", "ovalbumin", "lysozyme",
                   "oeufs", "uovo"],
    "Fish":       ["fish", "salmon", "tuna", "anchovy", "anchovies", "cod", "tilapia",
                   "sardine", "sardines", "mackerel", "halibut", "trout", "bass",
                   "flounder", "snapper", "haddock", "pollock", "herring", "swordfish",
                   "mahi", "sole", "pike", "carp", "surimi", "dace"],
    "Crustacean": ["shrimp", "lobster", "crab", "prawn", "prawns", "scallop", "scallops",
                   "clam", "clams", "oyster", "oysters", "crayfish", "crustacean",
                   "shellfish", "squid", "octopus", "mussel", "barnacle", "krill"],
    "TreeNuts":   ["almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
                   "pecan", "pecans", "pistachio", "pistachios", "hazelnut", "hazelnuts",
                   "brazil nut", "macadamia", "pine nut", "tree nut", "noisette",
                   "noix", "amande", "pistache", "anacarde"],
    "Peanuts":    ["peanut", "peanuts", "peanut oil", "peanut butter", "peanut flour",
                   "groundnut", "groundnuts", "arachis oil", "cacahuète", "cacahuete"],
    "Wheat":      ["wheat", "wheat flour", "semolina", "gluten", "barley", "rye",
                   "spelt", "durum", "wheat starch", "wheat bran", "enriched flour",
                   "unbleached flour", "whole wheat", "farine de blé", "froment",
                   "kamut", "triticale", "bulgur", "farro"],
    "Soybeans":   ["soy", "soya", "soybean", "soybeans", "soy sauce", "soy lecithin",
                   "soy protein", "tofu", "edamame", "miso", "tempeh", "whey protein",
                   "soja", "lecithine de soja", "tamari"],
    "Sesame":     ["sesame", "tahini", "sesame oil", "sesame seeds", "gingelly", "til",
                   "sésame", "sésamo"],
}

# Per spec Section 6 deduction table
DEDUCTIONS = {
    "direct":          85,
    "contains_stmt":   85,
    "may_contain":     45,
    "shared_facility": 35,
    "ambiguous_each":  10,
    "missing_data":    50,
}

CONFIDENCE_MODIFIERS = {"High": 0, "Medium": 8, "Low": 18}

AMBIGUOUS_TERMS = [
    "natural flavors", "natural flavor", "natural flavouring", "natural flavourings",
    "spices", "seasoning", "flavoring", "flavorings", "enzyme blend", "enzymes",
    "natural color", "extractives", "aroma", "arôme", "épices",
]

CROSS_CONTAMINATION_PHRASES = [
    "may contain", "might contain", "could contain", "produced in a facility",
    "shared equipment", "manufactured in", "processed in a facility",
    "made on equipment", "traces of", "peut contenir",
]


def _find_triggers(text_lower: str, allergen: str) -> list:
    return [kw for kw in ALLERGEN_KEYWORDS[allergen] if kw in text_lower]


def score_one(ingredients_text: str, allergen: str) -> dict:
    """
    Compute v1 deterministic score for a single allergen.
    Returns {score, confidence, triggers, ambiguity_count, rule_deduction,
             has_may_contain, has_cross_contamination}
    Score 0–100. Lower = more unsafe.
    """
    if not ingredients_text or not str(ingredients_text).strip():
        return {
            "score": max(0, 100 - DEDUCTIONS["missing_data"] - CONFIDENCE_MODIFIERS["Low"]),
            "confidence": "Low",
            "triggers": [],
            "ambiguity_count": 0,
            "rule_deduction": DEDUCTIONS["missing_data"],
            "has_may_contain": False,
            "has_cross_contamination": False,
        }

    text = str(ingredients_text).lower()
    triggers = _find_triggers(text, allergen)

    has_may_contain = any(p in text for p in CROSS_CONTAMINATION_PHRASES[:4])
    has_cross_cont  = any(p in text for p in CROSS_CONTAMINATION_PHRASES)

    # Determine deduction from strongest signal
    if triggers:
        if any(p in text for p in ["contains:", "allergens:", "contains "]):
            deduction = DEDUCTIONS["contains_stmt"]
        elif has_may_contain:
            deduction = DEDUCTIONS["may_contain"]
        elif has_cross_cont:
            deduction = DEDUCTIONS["shared_facility"]
        else:
            deduction = DEDUCTIONS["direct"]
    else:
        if has_may_contain:
            deduction = DEDUCTIONS["may_contain"] // 2  # Unconfirmed partial signal
        elif has_cross_cont:
            deduction = DEDUCTIONS["shared_facility"] // 2
        else:
            deduction = 0

    ambiguity_count = sum(1 for t in AMBIGUOUS_TERMS if t in text)

    confidence = (
        "Low"    if not str(ingredients_text).strip()
        else "Medium" if ambiguity_count >= 3
        else "High"
    )

    conf_mod = CONFIDENCE_MODIFIERS[confidence]
    score = max(0, min(100, 100 - deduction - (ambiguity_count * 5) - conf_mod))

    return {
        "score":                 score,
        "confidence":            confidence,
        "triggers":              triggers,
        "ambiguity_count":       ambiguity_count,
        "rule_deduction":        deduction,
        "has_may_contain":       has_may_contain,
        "has_cross_contamination": has_cross_cont,
    }


def score_all(ingredients_text: str) -> dict:
    """Score all 9 allergens for one product."""
    return {allergen: score_one(ingredients_text, allergen) for allergen in ALLERGEN_KEYWORDS}
