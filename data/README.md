# `data/` — bring your own

**No real food data ships with this repository.** What is tracked here is a
generator and the synthetic files it writes, so that the input contract is
documented by example and a smoke test runs on a fresh clone:

```
data/
├── make_example_data.py          # writes the three files below
├── example/
│   ├── food_df.example.csv       # 240 synthetic products, 30 columns
│   ├── thesaurus.example.xlsx    # 34 surface-form -> descriptor mappings
│   └── ingnut_df.example.csv     # 20 ingredients x 25 nutrients
└── README.md
```

The example numbers are drawn from arbitrary distributions and carry **no
nutritional meaning**. They exist to exercise the code path, not to produce a
result. Regenerate them with `python data/make_example_data.py`.

---

## The three inputs

Every entry point takes its data by argument; nothing is hard-coded to a path.

| Flag | What it is | Example file |
|---|---|---|
| `--food-df` | One row per packaged food: the label panel, the ingredient string, and precomputed UMAP embedding columns | `data/example/food_df.example.csv` |
| `--thesaurus-df` | Maps surface ingredient spellings to preferred descriptors | `data/example/thesaurus.example.xlsx` |
| `--ingnut-df` | Reference nutrient profile per preferred descriptor, used by the CVXPY optimizer | `data/example/ingnut_df.example.csv` |

### `--food-df`

Identifier and text columns: `fdc_id`, `brand_owner`, `market_country`,
`branded_food_category`, `brand_name`, `ingredients`, `serving_size`,
`serving_size_unit`, `description` and `publication_date`. No train/test
column is needed: `run.py` and the feature builders compute the split
themselves (`nutpred/split.py`), and a `data_type` column is ignored.

The eight panel macronutrients the model always conditions on (`nut8`):
`Energy(kcal)`, `Carbohydrate(g)`, `Total fat(g)`, `Protein(g)`, `Sodium(mg)`,
`Total sugar(g)`, `Saturated fatty acids(g)`, `Cholesterol(mg)`.

The three prediction targets: `Calcium(mg)`, `Fiber(g)`, `Iron(mg)`.

Carried through but not required by the core model: `Trans fatty acids(g)`,
`Potassium, K`, `Vitamin A`, `Vitamin A, IU`, `Vitamin A, RAE`,
`Vitamin C, total ascorbic acid`.

Embedding columns `umap_2`, `umap_10`, `umap_20`. Each cell is a
**string holding a JSON-style list** of that many floats, e.g.
`"[5.0385236, 5.0887556]"` for `umap_2`. The pipeline expands these into
`umap_20_1 … umap_20_20` etc.; it does **not** fit UMAP itself. Build them with
`python -m nutpred.build_ingredient_features --embed` (see "Regenerating the inputs").

All nutrient values are **per 100 g**. `serving_size` is numeric and
`serving_size_unit` is one of `g`/`grm`/`gm`/`ml`/`mlt` for the per-serving
label-rounding rules to apply; any other unit falls back to an exact-zero test.

### `--thesaurus-df`

Three columns: `Broad group`, `Preferred descriptor`, `Parsed ingredient term`.
One row per surface spelling. `Parsed ingredient term` is matched
case-insensitively against tokens split out of the `ingredients` string;
`Preferred descriptor` is the normalized name that must also appear in the
`ing` column of `--ingnut-df`.

The real analysis used the USDA Ingredient Thesaurus (Ahuja et al., 2023).

### `--ingnut-df`

`Rank`, `ing`, `case`, then 25 nutrient columns in SR Legacy naming:
`Energy`, `Water`, `Protein`, `Total lipid (fat)`,
`Carbohydrate, by difference`, `Sugars, total`, `Fiber, total dietary`,
`Calcium, Ca`, `Iron, Fe`, `Potassium, K`, `Sodium, Na`, `Zinc, Zn`,
`Vitamin A, RAE`, `Vitamin A, IU`, `Vitamin D`,
`Vitamin E (alpha-tocopherol)`, `Vitamin C, total ascorbic acid`, `Thiamin`,
`Riboflavin`, `Niacin`, `Cholesterol`, `Fatty acids, total trans`,
`Fatty acids, total saturated`, `Fatty acids, total monounsaturated`,
`Fatty acids, total polyunsaturated`.

`ing` must match the thesaurus's `Preferred descriptor`. `Rank` orders the
ingredient vocabulary by corpus frequency and determines the top-K cut; the
paper's pool is the top 100. Values are per 100 g of the ingredient.

---

## Regenerating the real inputs

The reported analysis used public, re-downloadable sources; none of them are
redistributed here. `docs/data_provenance_methods.md` records exactly which
releases were used and how they were retrieved.

1. **USDA FoodData Central, Branded Foods** — bulk CSV export
   (`food.csv`, `branded_food.csv`, `food_nutrient.csv`, `nutrient.csv`) from
   <https://fdc.nal.usda.gov/download-datasets/>. Join on `fdc_id`, keep the
   latest record per `gtin_upc` by `publication_date`, restrict to the snack
   categories, and assemble the columns listed above. The cohort-construction
   code is not distributed; the steps the reported cohort went through are in
   `docs/data_provenance_methods.md`, which also notes that the exact release
   used (December 2024) is an interim one FoodData Central no longer serves —
   the archived 2024-10-31 and 2025-04-24 releases bracket it.

2. **USDA SR Legacy** (April 2018, frozen) — the reference pool behind
   `--ingnut-df`.

3. **USDA Ingredient Thesaurus** — the controlled vocabulary behind
   `--thesaurus-df`.

4. **Ingredient features** — once the table above exists, one command adds all
   three ingredient-derived feature groups:

   ```bash
   python -m nutpred.build_ingredient_features --all \
     --in-csv  your_snack_table.csv \
     --thesaurus-xlsx your_thesaurus.xlsx \
     --top-k 100 --vocab-scope train \
     --out-csv your_model_input.csv
   ```

   `--binary` / `--score` / `--embed` select the groups individually.
   `--embed` calls the OpenAI embeddings API (cached to
   `data/embedding_cache`, created on demand, so re-runs are free) and reduces to 20 dimensions;
   `--embed-dims` changes that.

   **Fitting uses train rows only.** Both `--embed` (UMAP) and
   `--vocab-scope train` (vocabulary) compute the same split `run.py` makes
   and fit on its train rows; `--embed` projects the other rows through the
   fitted reducer. The split is saved as `SampleType`. `--embed` needs your
   own OpenAI key on a cache miss: `--api-key-file <file>` or
   `OPENAI_API_KEY`. `--vocab-scope train` also
   reports how far the train-only vocabulary diverges from an all-rows one.
