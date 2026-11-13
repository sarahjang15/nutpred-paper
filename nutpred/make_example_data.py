"""Draw the small runnable example in ``data/`` from the reported snack cohort.

The example is a real sample, not synthetic data: real products, real ingredient
statements, real label panels and real calcium/fiber/iron values, carrying their
USDA ``fdc_id`` so any row can be checked against FoodData Central. It exists to
document the input contract and to let ``python run.py`` work on a fresh clone.

**It is not a result and must never be read as one.** 104 products cannot
reproduce anything in the paper, and the draw is deliberately balanced rather
than representative: see "Sampling" below.

Sampling
--------
Products are stratified on the zero pattern of the three targets -- which of
calcium, iron and fiber are declared as zero -- the same eight cells the paper's
train/test split stratifies on. An equal quota is taken from each cell, so every
pattern appears and each target ends up about half zero and half non-zero.

That is on purpose. In the real cohort fiber is zero for only 5.5% of products,
so an unstratified 104-row draw leaves one or two fiber zeros per cross-
validation fold and the hurdle's Stage 1 classifier degenerates. Balancing the
cells exercises both stages for all three nutrients. The cost is that the zero
rates in this file -- and therefore anything computed from it -- say nothing
about the cohort's composition.

The embedding columns are the paper's: ``umap_20`` is carried over from the
reported model input, and ``umap_2``/``umap_10`` are produced by fitting UMAP on
the cohort's training rows with the reported settings and projecting the sample
through it, so the sample sits on the same manifold as the paper's embedding.

The thesaurus and the ingredient-nutrient reference table are not sampled at
all: both are copied whole, because both are small and both are the artifacts
the paper itself used. The thesaurus is copied byte for byte, so its checksum
matches the USDA release it came from.

Running it
----------
This script cannot run on a fresh clone: it needs the 14,019-product cohort,
which is not distributed. The exact invocation that produced the tracked files
is recorded in ``data/README.md``::

    python -m nutpred.make_example_data \
      --cohort     <snack_input_0222_fixed_umap20train.csv> \
      --embeddings <snack_input_model_compare_0222_fixed.csv> \
      --thesaurus  <THESAURUSFORPUBLICRELEASE.XLSX> \
      --ingnut     <ingnut_df_top100_rankfixed.csv>
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data"

TARGETS = ["Calcium(mg)", "Iron(mg)", "Fiber(g)"]
NUT8 = ["Energy(kcal)", "Carbohydrate(g)", "Total fat(g)", "Protein(g)",
        "Sodium(mg)", "Total sugar(g)", "Saturated fatty acids(g)", "Cholesterol(mg)"]
META = ["fdc_id", "brand_owner", "market_country", "branded_food_category",
        "brand_name", "description", "ingredients", "serving_size",
        "serving_size_unit", "publication_date"]

PER_CELL = 13          # 8 zero-pattern cells x 13 = 104 products
SEED = 42
UMAP_KW = dict(n_neighbors=30, min_dist=0.05, random_state=SEED)


def zero_pattern(df: pd.DataFrame) -> pd.Series:
    """'Z' where the target is declared zero, '.' otherwise, in Ca/Fe/Fiber order."""
    flags = (df[TARGETS] == 0)
    return flags.apply(lambda row: "".join("Z" if v else "." for v in row), axis=1)


def stratified_sample(cohort: pd.DataFrame) -> pd.DataFrame:
    cells = zero_pattern(cohort)
    picked = []
    for cell, idx in cells.groupby(cells).groups.items():
        pool = cohort.loc[idx]
        if len(pool) < PER_CELL:
            raise SystemExit(f"zero-pattern cell {cell!r} has only {len(pool)} products")
        picked.append(pool.sample(PER_CELL, random_state=SEED))
    out = pd.concat(picked).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def fit_umap(train_vecs: np.ndarray, sample_vecs: np.ndarray, dims: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_components=dims, **UMAP_KW)
    reducer.fit(train_vecs)
    return reducer.transform(sample_vecs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, required=True,
                    help="reported model input: metadata, panel, targets, umap_20, SampleType")
    ap.add_argument("--embeddings", type=Path, required=True,
                    help="table carrying fdc_id and ingredients_embedding")
    ap.add_argument("--thesaurus", type=Path, required=True, help="USDA Ingredient Thesaurus")
    ap.add_argument("--ingnut", type=Path, required=True, help="top-100 reference nutrient pool")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    for flag, path in [("--cohort", args.cohort), ("--embeddings", args.embeddings),
                       ("--thesaurus", args.thesaurus), ("--ingnut", args.ingnut)]:
        if not path.exists():
            sys.exit(f"{flag}: {path} not found. This script needs the reported cohort, "
                     f"which is not distributed with the repository.")

    cohort = pd.read_csv(args.cohort, dtype={"fdc_id": str}, low_memory=False)
    print(f"cohort: {len(cohort):,} products")

    emb = pd.read_csv(args.embeddings, usecols=["fdc_id", "ingredients_embedding"],
                      dtype={"fdc_id": str})
    emb = emb.dropna(subset=["ingredients_embedding"]).set_index("fdc_id")["ingredients_embedding"]
    def vecs(ids):
        return np.asarray([ast.literal_eval(emb.loc[i]) for i in ids], dtype=float)

    # A few hundred products carry no ingredient embedding. They cannot be
    # projected, and food_df is documented as arriving with umap_* already
    # computed, so they are outside the pool this example is drawn from.
    has_emb = cohort.fdc_id.isin(emb.index)
    print(f"  {int((~has_emb).sum())} products have no ingredient embedding and are excluded")
    cohort = cohort[has_emb].reset_index(drop=True)

    sample = stratified_sample(cohort)
    print(f"sampled: {len(sample)} products, {PER_CELL} per zero-pattern cell (seed {SEED})")

    train_ids = cohort.loc[cohort.SampleType == "train", "fdc_id"]
    train_vecs = vecs(train_ids)
    sample_vecs = vecs(sample.fdc_id)
    print(f"embeddings: fitting UMAP on {len(train_vecs):,} training rows, dim {train_vecs.shape[1]}")

    food = sample[META + NUT8 + TARGETS].copy()
    food["umap_20"] = sample["umap_20"].to_numpy()          # the paper's own projection
    for dims in (2, 10):
        proj = fit_umap(train_vecs, sample_vecs, dims)
        # .tolist() gives plain Python floats; list(ndarray) leaves np.float64
        # objects whose repr round-trips as "np.float64(...)" and is unparseable.
        food[f"umap_{dims}"] = [np.round(v, 6).tolist() for v in proj]
        print(f"  umap_{dims}: fitted and projected")
    food = food[META + NUT8 + TARGETS + ["umap_2", "umap_10", "umap_20"]]

    ingnut = pd.read_csv(args.ingnut)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    food.to_csv(args.out_dir / "food_df.example.csv", index=False)
    # Copied, not rewritten through pandas, so the shipped file is byte for byte
    # the USDA release and its checksum can be compared against the download.
    shutil.copyfile(args.thesaurus, args.out_dir / "thesaurus.xlsx")
    thes = pd.read_excel(args.out_dir / "thesaurus.xlsx")
    ingnut.to_csv(args.out_dir / "ingnut_df.example.csv", index=False)

    print(f"\nwrote {args.out_dir/'food_df.example.csv'}      {food.shape[0]} rows x {food.shape[1]} cols")
    print(f"wrote {args.out_dir/'thesaurus.xlsx'}           {thes.shape[0]} rows x {thes.shape[1]} cols (verbatim copy)")
    print(f"wrote {args.out_dir/'ingnut_df.example.csv'}    {ingnut.shape[0]} rows x {ingnut.shape[1]} cols")
    print("\nzeros per target in the sample (balanced by construction, NOT the cohort rate):")
    for t in TARGETS:
        print(f"  {t:12s} {int((food[t] == 0).sum()):3d} / {len(food)}")

    # Read the file back the way the pipeline will. A list of numpy scalars
    # serializes as "[np.float64(1.0), ...]", which pandas writes happily and
    # nothing can parse, so check the round trip rather than trust the write.
    reread = pd.read_csv(args.out_dir / "food_df.example.csv")
    for col in ("umap_2", "umap_10", "umap_20"):
        widths = {len(ast.literal_eval(cell)) for cell in reread[col]}
        if len(widths) != 1:
            raise SystemExit(f"{col}: ragged vectors {sorted(widths)}")
        print(f"  {col} round-trips as {widths.pop()} plain floats")


if __name__ == "__main__":
    main()
