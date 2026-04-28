"""
PuroScore v2 — Main Entry Point
Trains the hybrid model on the dataset, then runs an interactive
consultation session for new products.

Usage:
    python main.py                        # interactive mode
    python main.py --demo                 # run 3 built-in demo products
    python main.py --train-only           # train + save model, no consult
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.ml_model  import PuroV2Model
from src.consult   import generate

DATA_PATH  = os.path.join(os.path.dirname(__file__), "test_dataset.xlsx")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "puro_v2.pkl")

DEMO_PRODUCTS = [
    (
        "Granola Bar",
        "OATS, HONEY, NATURAL FLAVORS, CHOCOLATE CHIPS, MAY CONTAIN MILK",
    ),
    (
        "Caesar Dressing",
        "SOYBEAN OIL, WATER, PARMESAN CHEESE, ANCHOVIES, EGG YOLK, VINEGAR, SALT, SPICES",
    ),
    (
        "Plain Rice Cakes",
        "WHOLE GRAIN BROWN RICE, SALT",
    ),
]


def load_or_train(force_retrain: bool = False) -> PuroV2Model:
    if not force_retrain and os.path.exists(MODEL_PATH):
        print(f"  Loading saved model from {MODEL_PATH}")
        return PuroV2Model.load(MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: Dataset not found at {DATA_PATH}")
        print("  Place test_dataset.xlsx in the project root and re-run.\n")
        sys.exit(1)

    df = pd.read_excel(DATA_PATH)
    print(f"  Dataset loaded — {len(df)} rows, columns: {df.columns.tolist()}")

    model = PuroV2Model()
    model.train(df, verbose=True)
    model.save(MODEL_PATH)
    return model


def run_consult(model: PuroV2Model, product_name: str, ingredients: str):
    scores = model.predict_product(product_name, ingredients)
    report = generate(product_name, ingredients, scores, model.eval_results)
    print(report)


def interactive(model: PuroV2Model):
    print("\n" + "═" * 68)
    print("  PuroScore v2  ·  Allergen Consultation")
    print("  Type 'quit' to exit")
    print("═" * 68 + "\n")

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
                print("  Please enter ingredient text.\n")
                continue

            print()
            run_consult(model, name, ingr)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\n  Session ended.\n")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--train-only",  action="store_true")
    parser.add_argument("--retrain",     action="store_true")
    args = parser.parse_args()

    model = load_or_train(force_retrain=args.retrain)

    if args.train_only:
        print("  Training complete.")
        return

    if args.demo:
        for name, ingr in DEMO_PRODUCTS:
            run_consult(model, name, ingr)
            print()
        return

    interactive(model)


if __name__ == "__main__":
    main()
