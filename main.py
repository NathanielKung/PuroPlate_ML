"""
PuroScore v2 — Main Entry Point
Dataset: allergen_dataset_v6.xlsx

Usage:
    python3 main.py                  # interactive consultation mode
    python3 main.py --demo           # 3 built-in demo products
    python3 main.py --train-only     # train + save, no consult prompt
    python3 main.py --retrain        # force retrain (ignore saved model)
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml_model import PuroV2Model
from src.consult  import generate

DATA_PATH  = os.path.join(os.path.dirname(__file__), "allergen_dataset_v6.xlsx")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "puro_v2.pkl")

DEMO_PRODUCTS = [
    (
        "Granola Bar",
        "oats, honey, natural flavors, chocolate chips, may contain milk, produced in "
        "a facility that also processes wheat and peanuts",
    ),
    (
        "Caesar Dressing",
        "soybean oil, water, parmesan cheese (milk), anchovies (fish), egg yolk, "
        "vinegar, salt, garlic, spices, natural flavors allergens: milk, eggs, fish, soy",
    ),
    (
        "PESTO alla GENOVESE",
        "sunflower oil, fresh basil 30%, cashew nuts, parmigiano reggiano pdo cheese 5% "
        "(milk), maize fibre, whey powder (milk), salt, milk protein, extra virgin olive oil, "
        "sugar, basil extract, natural flavourings (milk), acidity regulator: lactic acid, "
        "garlic allergens: milk, nuts",
    ),
]


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: Dataset not found at {DATA_PATH}")
        print("  Place allergen_dataset_v6.xlsx in the project root.\n")
        sys.exit(1)

    # Row 0 = group headers (Product Info / CONTAINS / MAY CONTAIN / FLAGS)
    # Row 1 = actual column names
    df = pd.read_excel(DATA_PATH, header=1)
    print(f"  Dataset loaded — {len(df)} rows  |  {len(df.columns)} columns")
    return df


def load_or_train(force_retrain: bool = False) -> PuroV2Model:
    if not force_retrain and os.path.exists(MODEL_PATH):
        print(f"  Loading saved model from {MODEL_PATH}")
        model = PuroV2Model.load(MODEL_PATH)
        # Re-check if model was trained on older dataset schema
        if not model._trained:
            print("  Saved model invalid — retraining…")
            return load_or_train(force_retrain=True)
        return model

    df    = load_dataset()
    model = PuroV2Model()
    model.train(df, verbose=True)
    model.save(MODEL_PATH)
    return model


def run_consult(model: PuroV2Model, product_name: str, label_description: str):
    scores = model.predict_product(product_name, label_description)
    report = generate(product_name, label_description, scores, model.eval_results)
    print(report)


def interactive(model: PuroV2Model):
    print("\n" + "═" * 72)
    print("  PuroScore v2  ·  Allergen Consultation")
    print("  Paste the full ingredient / label text when prompted.")
    print("  Type 'quit' to exit.")
    print("═" * 72 + "\n")

    while True:
        try:
            name = input("  Product name  : ").strip()
            if name.lower() in ("quit", "exit", "q"):
                print("\n  Session ended.\n")
                break
            if not name:
                continue

            ingr = input("  Ingredients   : ").strip()
            if not ingr:
                print("  Please enter ingredient / label text.\n")
                continue

            print()
            run_consult(model, name, ingr)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\n  Session ended.\n")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",       action="store_true", help="Run 3 demo products")
    parser.add_argument("--train-only", action="store_true", help="Train + save, no prompt")
    parser.add_argument("--retrain",    action="store_true", help="Force retrain")
    args = parser.parse_args()

    model = load_or_train(force_retrain=args.retrain)

    if args.train_only:
        print("  Done.")
        return

    if args.demo:
        for name, ingr in DEMO_PRODUCTS:
            run_consult(model, name, ingr)
            print()
        return

    interactive(model)


if __name__ == "__main__":
    main()
