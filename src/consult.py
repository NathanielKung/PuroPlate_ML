"""
PuroScore v2 — Consultation Report Generator
Updated for v6 allergen names and richer output (Contains vs MayContain split).

Per spec: PuroScore is product-level only.
Red / Yellow / Green belongs to the PuroStatus (user-level) layer — not here.
"""

from datetime import date

ALLERGEN_DISPLAY = {
    "Milk":       "Milk (Dairy)",
    "Eggs":       "Eggs",
    "Fish":       "Fish",
    "Crustacean": "Crustacean Shellfish",
    "TreeNuts":   "Tree Nuts",
    "Peanuts":    "Peanuts",
    "Wheat":      "Wheat",
    "Soybeans":   "Soybeans",
    "Sesame":     "Sesame",
}

W = 72  # report width


def _bar(score: float, width: int = 18) -> str:
    filled = round((score / 100) * width)
    return "█" * filled + "░" * (width - filled)


def _risk_label(final_score: float, unsafe: bool, has_may_contain: bool) -> str:
    if not unsafe:
        return "✓ SAFE"
    if final_score < 25:
        return "⚠ HIGH RISK"
    elif final_score < 50:
        return "~ CAUTION"
    else:
        return "ℹ LOW RISK"


def _signal_type(d: dict) -> str:
    """Describe the primary signal type for the allergen."""
    if d["triggers"] and not d["has_may_contain"]:
        return "CONTAINS"
    elif d["triggers"] and d["has_may_contain"]:
        return "CONTAINS + MAY CONTAIN"
    elif d["has_may_contain"]:
        return "MAY CONTAIN"
    elif d["ambiguity"] > 0:
        return f"AMBIGUOUS ({d['ambiguity']} flag{'s' if d['ambiguity']>1 else ''})"
    return "—"


def generate(product_name: str, label_description: str, scores: dict,
             eval_results: dict | None = None) -> str:
    lines = []

    def hr(c="═"):  return c * W
    def rule(c="─"): return c * W

    lines += [
        hr(),
        "  PUROPLATE ALLERGEN CONSULTATION REPORT".center(W),
        "  PuroScore v2  ·  Hybrid Rules + Logistic Regression".center(W),
        hr(),
        f"  Product     : {product_name}",
        f"  Ingredients : {label_description[:W - 16]}{'…' if len(label_description) > W - 16 else ''}",
        f"  Analyzed    : {date.today().isoformat()}",
        rule(),
        "  ALLERGEN RISK PROFILE",
        rule(),
        "",
        f"  {'Allergen':<24} {'Score':>5}  {'ML Risk':>7}  {'Conf':>6}  {'Signal Type':<22}  Status",
        "  " + "─" * (W - 2),
    ]

    risky, safe = [], []

    for allergen, label in ALLERGEN_DISPLAY.items():
        if allergen not in scores:
            continue
        d = scores[allergen]
        status = _risk_label(d["final_score"], d["unsafe"], d["has_may_contain"])
        sig    = _signal_type(d)
        ml_pct = f"{d['ml_prob'] * 100:.0f}%"

        lines.append(
            f"  {label:<24} {d['final_score']:>5.1f}  {ml_pct:>7}  "
            f"{d['confidence']:>6}  {sig:<22}  {status}"
        )

        if d["unsafe"]:
            risky.append((allergen, label, d))
        else:
            safe.append(label)

    lines += ["", rule()]

    # ── Triggers breakdown ────────────────────────────────────────────────────
    if risky:
        lines += ["  TRIGGER BREAKDOWN", rule(), ""]
        for _, label, d in risky:
            lines.append(f"  {label}")
            if d["triggers"]:
                lines.append(f"    Direct ingredients  : {', '.join(d['triggers'][:5])}")
            if d["has_may_contain"]:
                lines.append(f"    Precautionary stmt  : 'may contain' or cross-contamination language detected")
            if d["ambiguity"] > 0:
                lines.append(f"    Ambiguous signals   : {d['ambiguity']} flag(s) — e.g. 'natural flavors', 'spices'")
            lines.append(f"    v1 baseline score   : {d['v1_score']}  →  ML risk: {d['ml_prob']*100:.0f}%  →  Final: {d['final_score']:.1f}")
            lines.append("")
        lines.append(rule())

    # ── Recommendation ────────────────────────────────────────────────────────
    lines += ["  RECOMMENDATION", rule(), ""]

    if risky:
        lines.append(f"  ⚠  ALLERGENS FLAGGED ({len(risky)})")
        for _, label, d in risky:
            sig = _signal_type(d)
            lines.append(
                f"     {label:<24} Score {d['final_score']:.1f}  |  "
                f"ML risk {d['ml_prob']*100:.0f}%  |  Conf: {d['confidence']}  |  {sig}"
            )
        lines += [
            "",
            "     ⚠  PuroScore is product-level only.",
            "        Pass these scores to the PuroStatus layer with the user's",
            "        severity profile (Severe / Moderate / Mild) to compute",
            "        Red / Yellow / Green status and gating behavior.",
        ]
    else:
        lines.append("  ✓  No allergens flagged for this product.")

    if safe:
        lines += [
            "",
            f"  ✓  CLEAR ({len(safe)})",
            "     " + ", ".join(safe),
        ]

    lines += ["", rule()]

    # ── Model metadata ────────────────────────────────────────────────────────
    lines += ["  MODEL METADATA", rule(), ""]
    lines += [
        "  Engine      : PuroScore v2  Hybrid (Rules + Logistic Regression)",
        "  Dataset     : allergen_dataset_v6  (Contains + MayContain split columns)",
        "  Layer       : 1 — Product-Level  |  No UI colors returned",
        "  Blend       : α=0.35 (v1 rules)  +  β=0.65 (ML probability)",
        "  Threshold   : Recall-optimised 0.30  |  Priority: no false negatives",
        "  Target      : unsafe = Contains OR MayContain per allergen",
    ]

    if eval_results:
        lines += ["", "  Per-allergen 5-fold CV:"]
        lines.append(
            f"  {'Allergen':<24} {'Recall':>7}  {'Prec':>7}  "
            f"{'Contains':>9}  {'MayCont':>8}  {'Unsafe':>7}  {'Total':>6}"
        )
        lines.append("  " + "─" * 68)
        for allergen, label in ALLERGEN_DISPLAY.items():
            if allergen in eval_results:
                r = eval_results[allergen]
                lines.append(
                    f"  {label:<24} {r['recall_mean']:>6.3f}   {r['precision_mean']:>6.3f}  "
                    f"{r['n_contains']:>9}  {r['n_may_contain']:>8}  "
                    f"{r['n_unsafe']:>7}  {r['n_total']:>6}"
                )

    lines += [
        "",
        hr(),
    ]

    return "\n".join(lines)
