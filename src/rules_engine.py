"""
PuroScore v1 — Deterministic Rules Engine
Layer 1 (Product-Level) baseline scorer.
Per the spec: rules = hard truth. No UI colors returned here.
"""

ALLERGEN_KEYWORDS = {
    "Milk":      ["milk", "dairy", "whey", "casein", "caseinate", "lactose", "butter",
                  "cream", "cheese", "milk powder", "milk solids", "parmesan", "cheddar",
                  "mozzarella", "buttermilk", "ghee", "yogurt", "skim milk", "whole milk",
                  "nonfat milk", "sodium caseinate"],
    "Egg":       ["egg", "eggs", "egg whites", "egg white", "egg yolk", "egg albumen",
                  "whole egg", "mayonnaise", "albumin", "ovalbumin"],
    "Peanut":    ["peanut", "peanuts", "peanut oil", "peanut butter", "peanut flour",
                  "groundnut", "groundnuts", "arachis oil"],
    "Tree_Nuts": ["almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
                  "pecan", "pecans", "pistachio", "pistachios", "hazelnut", "hazelnuts",
                  "brazil nut", "macadamia", "pine nut", "tree nut", "coconut"],
    "Soy":       ["soy", "soya", "soybean", "soybeans", "soy sauce", "soy lecithin",
                  "soy protein", "tofu", "edamame", "miso", "tempeh", "whey protein"],
    "Wheat":     ["wheat", "wheat flour", "semolina", "gluten", "barley", "rye",
                  "spelt", "durum", "wheat starch", "wheat bran", "enriched flour",
                  "unbleached flour", "whole wheat"],
    "Fish":      ["fish", "salmon", "tuna", "anchovy", "anchovies", "cod", "tilapia",
                  "sardine", "sardines", "mackerel", "halibut", "trout", "bass",
                  "flounder", "snapper", "haddock"],
    "Shellfish": ["shrimp", "lobster", "crab", "prawn", "prawns", "scallop", "scallops",
                  "clam", "clams", "oyster", "oysters", "crayfish", "crustacean",
                  "shellfish", "squid", "octopus"],
    "Sesame":    ["sesame", "tahini", "sesame oil", "sesame seeds", "gingelly", "til"],
}

AMBIGUOUS_TERMS = [
    "natural flavors", "natural flavor", "spices", "flavoring", "flavorings",
    "enzyme blend", "enzymes", "natural color", "extractives", "seasoning",
]

# Deduction table from spec Section 6
DEDUCTIONS = {
    "direct":            85,
    "contains_stmt":     85,
    "may_contain":       45,
    "shared_facility":   35,
    "ambiguous_each":    10,
    "missing_data":      50,
}

CONFIDENCE_MODIFIERS = {"High": 0, "Medium": 8, "Low": 18}


def _find_triggers(text_lower: str, allergen: str) -> list[str]:
    return [kw for kw in ALLERGEN_KEYWORDS[allergen] if kw in text_lower]


def score_one(ingredients_text: str, allergen: str) -> dict:
    """
    Compute v1 deterministic score for a single allergen.
    Returns: {score, confidence, triggers, ambiguity_count, rule_deduction}
    Score range 0–100. Lower = more unsafe.
    """
    if not ingredients_text or not str(ingredients_text).strip():
        return {
            "score": max(0, 100 - DEDUCTIONS["missing_data"] - CONFIDENCE_MODIFIERS["Low"]),
            "confidence": "Low",
            "triggers": [],
            "ambiguity_count": 0,
            "rule_deduction": DEDUCTIONS["missing_data"],
        }

    text = str(ingredients_text).lower()
    triggers = _find_triggers(text, allergen)

    # Determine max single-signal deduction
    if triggers:
        if "contains" in text and any(kw in text for kw in triggers):
            deduction = DEDUCTIONS["contains_stmt"]
        elif "may contain" in text and any(kw in text for kw in triggers):
            deduction = DEDUCTIONS["may_contain"]
        elif "facility" in text or "equipment" in text:
            deduction = DEDUCTIONS["shared_facility"]
        else:
            deduction = DEDUCTIONS["direct"]
    else:
        if "may contain" in text:
            deduction = DEDUCTIONS["may_contain"] // 2   # Unconfirmed partial signal
        else:
            deduction = 0

    ambiguity_count = sum(1 for t in AMBIGUOUS_TERMS if t in text)

    # Confidence tier
    if not str(ingredients_text).strip():
        confidence = "Low"
    elif ambiguity_count >= 3:
        confidence = "Medium"
    else:
        confidence = "High"

    conf_mod = CONFIDENCE_MODIFIERS[confidence]
    score = max(0, min(100, 100 - deduction - (ambiguity_count * 5) - conf_mod))

    return {
        "score": score,
        "confidence": confidence,
        "triggers": triggers,
        "ambiguity_count": ambiguity_count,
        "rule_deduction": deduction,
    }


def score_all(ingredients_text: str) -> dict:
    """Score all 9 allergens for one product. Returns dict keyed by allergen name."""
    return {allergen: score_one(ingredients_text, allergen) for allergen in ALLERGEN_KEYWORDS}
