#!/usr/bin/env python3
"""Generate the synthetic example inputs in ``data/example/``.

No real food data ships with this repository. What ships instead is a small
synthetic cohort with exactly the columns the pipeline reads, so that

  * the ``--food-df`` / ``--thesaurus-df`` / ``--ingnut-df`` contract is
    self-documenting -- the schema is the file, not a paragraph in a README --
    and
  * ``python run.py --smoke-test-rows 100`` runs end to end on a fresh clone
    with no downloads.

The numbers are drawn from arbitrary distributions and carry no nutritional
meaning whatsoever. Do not use them for anything but exercising the code path.
To reproduce the paper you must supply the real inputs; see data/README.md and docs/data_provenance_methods.md.

Usage:
    python data/make_example_data.py [--rows 240] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "example"

# The eight macronutrients the model always has (nut8), plus the three targets
# and the extra panel columns carried through from FoodData Central.
NUT8 = [
    "Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)",
    "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)",
]
EXTRA_PANEL = [
    "Trans fatty acids(g)", "Potassium, K", "Vitamin A", "Vitamin A, IU",
    "Vitamin A, RAE", "Vitamin C, total ascorbic acid",
]
UMAP_DIMS = [2, 5, 10, 20]

# A toy ingredient vocabulary. The thesaurus maps surface forms in the
# ``ingredients`` string onto these preferred descriptors, and the ingnut table
# gives each one a nutrient profile -- the same three-file relationship the
# real inputs have.
VOCAB = [
    "WHEAT FLOUR", "SUGAR", "SALT", "VEGETABLE OIL", "CORN STARCH",
    "MILK, NONFAT, DRIED", "BUTTER", "COCOA", "RICE FLOUR", "POTATO FLOUR",
    "YEAST", "SOY LECITHIN", "BAKING SODA", "OAT FIBER", "WHEY PROTEIN",
    "EGG", "HONEY", "CINNAMON", "TOMATO POWDER", "SPINACH POWDER",
]
# Surface spellings a label might actually use, per preferred descriptor.
SURFACE = {
    "WHEAT FLOUR": ["wheat flour", "enriched wheat flour", "whole wheat flour"],
    "SUGAR": ["sugar", "cane sugar", "organic sugar"],
    "SALT": ["salt", "sea salt"],
    "VEGETABLE OIL": ["vegetable oil", "expeller pressed sunflower oil"],
    "CORN STARCH": ["corn starch", "cornstarch"],
    "MILK, NONFAT, DRIED": ["nonfat dry milk", "nonfat milk powder"],
    "BUTTER": ["butter", "sweet cream butter"],
    "COCOA": ["cocoa", "cocoa powder"],
    "RICE FLOUR": ["rice flour"],
    "POTATO FLOUR": ["potato flour"],
    "YEAST": ["yeast"],
    "SOY LECITHIN": ["soy lecithin", "lecithin"],
    "BAKING SODA": ["baking soda", "sodium bicarbonate"],
    "OAT FIBER": ["oat fiber"],
    "WHEY PROTEIN": ["whey protein", "whey protein concentrate"],
    "EGG": ["egg", "whole egg"],
    "HONEY": ["honey"],
    "CINNAMON": ["cinnamon"],
    "TOMATO POWDER": ["tomato powder"],
    "SPINACH POWDER": ["spinach powder"],
}
CATEGORIES = [
    "Cookies & Biscuits", "Chips, Pretzels & Snacks", "Breakfast Cereals",
    "Crackers", "Snack, Energy & Granola Bars",
]

# ingnut_df columns, in the order the real table carries them.
INGNUT_COLS = [
    "Rank", "ing", "case", "Energy", "Water", "Protein", "Total lipid (fat)",
    "Carbohydrate, by difference", "Sugars, total", "Fiber, total dietary",
    "Calcium, Ca", "Iron, Fe", "Potassium, K", "Sodium, Na", "Zinc, Zn",
    "Vitamin A, RAE", "Vitamin A, IU", "Vitamin D",
    "Vitamin E (alpha-tocopherol)", "Vitamin C, total ascorbic acid",
    "Thiamin", "Riboflavin", "Niacin", "Cholesterol",
    "Fatty acids, total trans", "Fatty acids, total saturated",
    "Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated",
]


def make_food_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        k = int(rng.integers(3, 11))
        picks = list(rng.choice(len(VOCAB), size=k, replace=False))
        ing_text = ", ".join(
            str(rng.choice(SURFACE[VOCAB[j]])) for j in picks
        ).upper() + "."
        rec: dict[str, object] = {
            "fdc_id": 1_000_000 + i,
            "brand_owner": f"Example Foods {i % 17}",
            "market_country": "United States",
            "branded_food_category": str(rng.choice(CATEGORIES)),
            "brand_name": f"BRAND{i % 23}",
            "ingredients": ing_text,
            "serving_size": float(np.round(rng.uniform(15, 60), 1)),
            "serving_size_unit": "g",
            "description": f"EXAMPLE SNACK PRODUCT {i}",
            "publication_date": "2024-01-01",
        }
        rec["Energy(kcal)"] = float(np.round(rng.uniform(80, 560), 1))
        rec["Carbohydrate(g)"] = float(np.round(rng.uniform(5, 85), 2))
        rec["Total fat(g)"] = float(np.round(rng.uniform(0, 40), 2))
        rec["Protein(g)"] = float(np.round(rng.uniform(0, 25), 2))
        rec["Sodium(mg)"] = float(np.round(rng.uniform(0, 1400), 1))
        rec["Total sugar(g)"] = float(np.round(rng.uniform(0, 55), 2))
        rec["Saturated fatty acids(g)"] = float(np.round(rng.uniform(0, 18), 2))
        rec["Cholesterol(mg)"] = float(np.round(rng.uniform(0, 90), 1))
        for c in EXTRA_PANEL:
            rec[c] = float(np.round(rng.uniform(0, 10), 2))
        # Targets are zero-inflated, which is the whole reason the hurdle
        # model exists -- the example data reproduces that shape so a
        # --hurdle smoke test exercises both stages.
        for c, hi, pz in (("Calcium(mg)", 400.0, 0.35),
                          ("Fiber(g)", 14.0, 0.30),
                          ("Iron(mg)", 8.0, 0.32)):
            rec[c] = 0.0 if rng.random() < pz else float(np.round(rng.uniform(0.1, hi), 2))
        for d in UMAP_DIMS:
            # .tolist() -> plain floats, "[1.23, ...]", the format the
            # pipeline writes and reads.
            rec[f"umap_{d}"] = str(np.round(rng.uniform(-8, 12, d), 6).tolist())
        rows.append(rec)
    cols = (["fdc_id", "brand_owner", "market_country", "branded_food_category",
             "brand_name", "ingredients", "serving_size", "serving_size_unit",
             "description", "publication_date"]
            + ["Calcium(mg)", "Carbohydrate(g)", "Cholesterol(mg)", "Energy(kcal)",
               "Saturated fatty acids(g)", "Trans fatty acids(g)", "Fiber(g)",
               "Iron(mg)", "Potassium, K", "Protein(g)", "Sodium(mg)",
               "Total sugar(g)", "Total fat(g)", "Vitamin A", "Vitamin A, IU",
               "Vitamin A, RAE", "Vitamin C, total ascorbic acid"]
            + [f"umap_{d}" for d in UMAP_DIMS])
    return pd.DataFrame(rows)[cols]


def make_thesaurus() -> pd.DataFrame:
    rows = []
    for pref in VOCAB:
        for surf in SURFACE[pref]:
            rows.append({
                "Broad group": "Example group",
                "Preferred descriptor": pref,
                "Parsed ingredient term": surf,
            })
    return pd.DataFrame(rows)


def make_ingnut(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for rank, ing in enumerate(VOCAB, start=1):
        rec: dict[str, object] = {"Rank": rank, "ing": ing, "case": np.nan}
        for c in INGNUT_COLS[3:]:
            rec[c] = float(np.round(rng.uniform(0, 100), 2))
        rec["Energy"] = float(np.round(rng.uniform(0, 900), 1))
        rec["Water"] = float(np.round(rng.uniform(0, 90), 1))
        rows.append(rec)
    return pd.DataFrame(rows)[INGNUT_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=240,
                    help="rows in the example food table (default 240)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    food = make_food_df(rng, args.rows)
    food.to_csv(OUT / "food_df.example.csv", index=False)

    thes = make_thesaurus()
    thes.to_excel(OUT / "thesaurus.example.xlsx", index=False)

    ingnut = make_ingnut(rng)
    ingnut.to_csv(OUT / "ingnut_df.example.csv", index=False)

    print(f"wrote {OUT/'food_df.example.csv'}       {food.shape[0]} rows x {food.shape[1]} cols")
    print(f"wrote {OUT/'thesaurus.example.xlsx'}    {thes.shape[0]} rows x {thes.shape[1]} cols")
    print(f"wrote {OUT/'ingnut_df.example.csv'}     {ingnut.shape[0]} rows x {ingnut.shape[1]} cols")


if __name__ == "__main__":
    main()
