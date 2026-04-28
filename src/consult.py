"""
PuroScore v2 — Consultation Report Generator
Outputs a clean, professional allergen consultation for a product.
Per spec: PuroScore returns numeric scores + confidence only.
Red/Yellow/Green (PuroStatus) is NOT computed here — that belongs to the user-layer.
"""

from datetime import date

ALLERGEN_DISPLAY = {
    "Milk":      "Milk (Dairy)",
    "Egg":       "Egg",
    "Peanut":    "Peanut",
    "Tree_Nuts": "Tree Nuts",
    "Soy":       "Soy",
    "Wheat":     "Wheat",
    "Fish":      "Fish",
    "Shellfish": "Crustacean Shellfish",
    "Sesame":    "Sesame",
}

W = 68  # report width


def _bar(score: float, width: int = 20) -> str:
    filled = round((score / 100) * width)
    return "█" * filled + "░" * (width - filled)


def _risk_label(final_score: float, unsafe: bool) -> str:
    if unsafe:
        if final_score < 30:
            return "⚠ HIGH RISK"
        elif final_score < 55:
            return "~ MODERATE"
        else:
            return "ℹ LOW RISK"
    return "✓ SAFE"


def generate(product_name: str, ingredients: str, scores: dict,
             eval_results: dict | None = None) -> str:
    """
    Render a full consultation report string.
    `scores` is the output of PuroV2Model.predict_product().
    """
    lines = []

    def rule(char="─"):  return char * W
    def hr(char="═"):    return char * W

    lines += [
        hr("═"),
        "  PUROPLATE ALLERGEN CONSULTATION REPORT".center(W),
        "  PuroScore v2  ·  Hybrid Rules + ML Engine".center(W),
        hr("═"),
        f"  Product     : {product_name}",
        f"  Ingredients : {ingredients[:W - 16]}{'…' if len(ingredients) > W - 16 else ''}",
        f"  Analyzed    : {date.today().isoformat()}",
        rule(),
        "  ALLERGEN RISK PROFILE",
        rule(),
        "",
        f"  {'Allergen':<22} {'Score':>5}  {'ML Risk':>7}  {'Conf':>6}  Bar (safe →)  Status",
        "  " + "─" * (W - 2),
    ]

    risky    = []
    safe     = []

    for allergen, label in ALLERGEN_DISPLAY.items():
        if allergen not in scores:
            continue
        d = scores[allergen]
        bar    = _bar(d["final_score"])
        status = _risk_label(d["final_score"], d["unsafe"])
        ml_pct = f"{d['ml_prob'] * 100:.0f}%"
        lines.append(
            f"  {label:<22} {d['final_score']:>5.1f}  {ml_pct:>7}  "
            f"{d['confidence']:>6}  {bar}  {status}"
        )
        if d["unsafe"]:
            risky.append((allergen, label, d))
        else:
            safe.append(label)

    lines += ["", rule()]

    # ── Triggers ──────────────────────────────────────────────────────────────
    if any(scores[a]["triggers"] or scores[a]["ambiguity"] > 0
           for a in scores if scores[a]["unsafe"]):
        lines += ["  DETECTED TRIGGERS", rule(), ""]
        for allergen, label, d in risky:
            if d["triggers"]:
                tlist = ", ".join(d["triggers"][:4])
                lines.append(f"  {label:<22} → Direct: {tlist}")
            if d["ambiguity"] > 0:
                lines.append(f"  {'':22}   Ambiguous terms: {d['ambiguity']} "
                             f"(e.g. natural flavors, spices)")
            if d["triggers"] or d["ambiguity"] > 0:
                lines.append("")
        lines.append(rule())

    # ── Summary ───────────────────────────────────────────────────────────────
    lines += ["  RECOMMENDATION", rule(), ""]

    if risky:
        lines.append(f"  ⚠  ALLERGENS FLAGGED ({len(risky)})")
        for allergen, label, d in risky:
            lines.append(f"     {label:<22} Score {d['final_score']:.1f}  "
                         f"ML risk {d['ml_prob']*100:.0f}%  Conf: {d['confidence']}")
        lines.append("")
        lines.append("     Note: Consult PuroStatus layer with user severity profile")
        lines.append("     to determine Red / Yellow / Green UI status.")
    else:
        lines.append("  ✓  No allergens flagged by v2 engine.")

    if safe:
        lines.append("")
        lines.append(f"  ✓  CLEAR ({len(safe)})")
        lines.append("     " + ", ".join(safe))

    lines += ["", rule()]

    # ── Model metadata ────────────────────────────────────────────────────────
    lines += ["  MODEL METADATA", rule(), ""]
    lines += [
        "  Engine      : PuroScore v2 — Hybrid (Rules + Logistic Regression)",
        "  Layer       : 1 — Product-Level only  |  No UI colors returned",
        "  Blend       : α=0.35 (v1 rules)  +  β=0.65 (ML probability)",
        "  Threshold   : Recall-optimised at 0.35 (minimises false negatives)",
        "  Training    : Binary Relevance — 1 calibrated classifier per allergen",
    ]

    if eval_results:
        lines += ["", "  Per-allergen CV Recall (5-fold):"]
        for allergen, label in ALLERGEN_DISPLAY.items():
            if allergen in eval_results:
                r = eval_results[allergen]
                lines.append(
                    f"    {label:<24} recall {r['recall_mean']:.3f} ±{r['recall_std']:.3f}"
                    f"  |  prec {r['precision_mean']:.3f}"
                    f"  |  positives {r['n_positive']}/{r['n_total']}"
                )

    lines += [
        "",
        "  ⚠  PuroScore is product-level. User-severity-based Red/Yellow/Green",
        "     status is determined separately by the PuroStatus engine.",
        "",
        hr("═"),
    ]

    return "\n".join(lines)
