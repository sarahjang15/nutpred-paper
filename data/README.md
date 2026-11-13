# `data/` — bring your own

**The reported cohort of 14,019 snack products is not distributed.** What is
tracked here is a small example drawn from it, together with the two reference
tables the analysis used in full, so that the input contract is documented by
example and a smoke test runs on a fresh clone:

```
data/
├── food_df.example.csv           # 104 real products, 24 columns
├── thesaurus.xlsx                # the USDA Ingredient Thesaurus, all 25,671 rows
├── ingnut_df.example.csv         # the paper's top-100 reference pool, 28 columns
├── figure2.png                   # ingredient-feature diagram, used by the main README
└── README.md
```

The products are real — real USDA `fdc_id`, ingredient statements, label panels
and calcium/fiber/iron values — so any row can be checked against FoodData
Central. The thesaurus and the reference table are not samples at all: they are
the paper's own, complete. All three USDA sources are public domain.

## This example is not a result

104 products reproduce nothing in the paper, and the draw is **deliberately
balanced, not representative**. Products are stratified on the zero pattern of
the three targets — which of calcium, iron and fiber are declared zero — the
same eight cells the paper's split stratifies on, with 13 taken from each. Each
target therefore comes out 52 zero / 52 non-zero.

That is on purpose. Fiber is zero for only 5.5% of the real cohort (calcium
42.1%, iron 20.2%), so an unstratified 104-row draw leaves one or two fiber
zeros per cross-validation fold and the hurdle's Stage 1 classifier
degenerates. Balancing the cells exercises both stages for all three nutrients.
The cost is that zero rates here, and anything computed from them, say nothing
about the cohort's composition.

What *is* representative: `thesaurus.xlsx` and `ingnut_df.example.csv` are the
paper's own tables, complete, so ingredient mapping and the reference profiles
behave here exactly as they do on the full cohort. `umap_20` is the paper's
train-fitted projection; `umap_2` and `umap_10` come from fitting UMAP on the
cohort's training rows with the reported settings (`n_neighbors=30`,
`min_dist=0.05`, seed 42) and projecting the sample through it.

## Regenerating it

`nutpred/make_example_data.py` draws the sample, but **it cannot run on a fresh
clone** — it needs the cohort, which is not distributed. The invocation that
produced the tracked files was:

```bash
python -m nutpred.make_example_data \
  --cohort     snack_input_0222_fixed_umap20train.csv \
  --embeddings snack_input_model_compare_0222_fixed.csv \
  --thesaurus  THESAURUSFORPUBLICRELEASE.XLSX \
  --ingnut     ingnut_df_top100_rankfixed.csv
```

Seed 42, 13 products per zero-pattern cell. Products carrying no ingredient
embedding (438 of 14,019) are outside the pool, since they cannot be projected.
Only the food table is sampled; the thesaurus is copied byte for byte, so
`sha256 data/thesaurus.xlsx` is `10be9300…2c6f2910`, the checksum of the USDA
release itself.

---

## The three inputs

Every entry point takes its data by argument; nothing is hard-coded to a path.

| Flag | What it is | Example file |
|---|---|---|
| `--food-df` | One row per packaged food: the label panel, the ingredient string, and precomputed UMAP embedding columns | `data/food_df.example.csv` |
| `--thesaurus-df` | Maps surface ingredient spellings to preferred descriptors | `data/thesaurus.xlsx` |
| `--ingnut-df` | Reference nutrient profile per preferred descriptor, used by the CVXPY optimizer | `data/ingnut_df.example.csv` |
| `--yf-column` | Name of a column in `--food-df` holding per-ingredient yield factors; only for `--method yf_opt` | written by `nutpred.predict_yield_factors` |

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

### `--yf-column` (only for `--method yf_opt`)

Not a file but a column of `--food-df`: for each product, a list of processing
yield factors, one per entry of that row's mapped ingredient list and in the same
order, serialized like the embedding columns (`"[0.95, 0.8, 1.0]"`). A yield factor
is the mass an ingredient keeps in the finished product over the mass that went in,
so 0 < YF <= 1; the optimizer divides that ingredient's reference profile by it.

Build the column with

```bash
python -m nutpred.predict_yield_factors \
  --food-df <your table> --thesaurus-df <your thesaurus> \
  --api-key-file <your key> --out-csv food_df_yf.csv
```

which estimates them with an LLM from the product description, brand, category,
declared base panel and ordered ingredient list, and writes `pred_yield_factors`.
The three target nutrients are never sent to the model. Responses are cached, so a
re-run is free. If a list's length does not match the row's mapped ingredient
count, that product silently falls back to unit yield factors with a warning, so
check the log when supplying the column yourself.

### `--thesaurus-df`

Three columns: `Broad group`, `Preferred descriptor`, `Parsed ingredient term`.
One row per surface spelling. `Parsed ingredient term` is matched
case-insensitively against tokens split out of the `ingredients` string;
`Preferred descriptor` is the normalized name that must also appear in the
`ing` column of `--ingnut-df`.

`data/thesaurus.xlsx` is the USDA Ingredient Thesaurus (Ahuja et al., 2023) as
the analysis used it: all 25,671 surface forms over 3,042 preferred descriptors
in 15 broad groups. Only a few hundred of those entries match any of the 104
example labels, but shipping it whole means your own food table maps against
the real vocabulary without downloading anything.

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
