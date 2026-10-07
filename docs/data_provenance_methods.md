# Data provenance: USDA FoodData Central download dates

Two USDA FoodData Central (FDC) products underlie this work: the **Branded Foods**
dataset (model input) and **SR Legacy** (the generic-ingredient nutrient reference).
This note records what each local copy is, when it was obtained, and where each lands
against the releases currently listed at <https://fdc.nal.usda.gov/download-datasets/>.

---

## 1. Paste-ready Methods text

### 2.1.1. USDA FoodData Central Global Branded Food Products Database

The Global Branded Food Products Database (USDA Agricultural Research Service,
FoodData Central; https://fdc.nal.usda.gov/download-datasets/) was used as the primary
source of branded food data. Information was extracted from the bulk comma-separated
export, specifically from food.csv, branded_food.csv, food_nutrient.csv and
nutrient.csv. These tables provided product identifiers, branded food metadata,
nutrient values and nutrient definitions, respectively. Tables were linked using
fdc_id. When multiple records corresponded to the same gtin_upc, the latest record was
retained according to publication_date. The export used here corresponded to the
December 2024 release of the Branded Foods dataset and comprised 1,981,655 branded
records with publication dates ranging from 1 April 2019 to 19 December 2024; it was
retrieved between December 2024 and 6 April 2025. Because FoodData Central archives the
Branded Foods dataset only at its semi-annual release points, the December 2024 release
is no longer distributed, and the archived releases of 31 October 2024 and 24 April 2025
bracket the version analysed here. Records published after 31 October 2024, which are
therefore absent from the nearest archived release, accounted for 5,014 of the 1,981,655
raw records (0.25%) and for 3,443 of the 292,908 records (1.18%) retained in the final
modelling dataset.

### 2.1.2. USDA SR Legacy

To support ingredient-aware inference, analytical nutrient profiles for commonly
occurring ingredients were curated from USDA SR Legacy Foods (April 2018 release,
retrieved November 2025). SR Legacy is the final release of the Standard Reference data
type and comprises 7,793 foods; it has not been revised since publication, so the
retrieval date does not affect the version of the data used. All 7,793 foods were
retained as the reference pool, and these profiles were used as reference nutrient
values in the ingredient-ratio optimization module.

### 2.1.3. USDA Ingredient Thesaurus

Ingredient normalization used the USDA Ingredient Thesaurus, which provides a controlled
vocabulary and mappings from commercial ingredient strings to standardized ingredient
terms (Ahuja et al., 2023).

*(Substitute the cohort figures the manuscript reports. For reference, the all-category
build measured here gives 292,908 records, 3,443 of them (1.18%) published after
31 October 2024; an earlier hand-assembled input gave 312,252 records, 4,291 (1.37%).
The reported analysis uses the snack subset of this population.)*

---

## 2. Evidence behind those dates

### Branded Foods

| Item | Value | Source |
|---|---|---|
| Raw export directory | local copy of the FDC Branded Foods bulk export | `food.csv`, `branded_food.csv`, `food_nutrient.csv`, `nutrient.csv` |
| Member-file mtimes | 2024-12-13, 17:26–17:51 UTC | `stat` on `food.csv`, `branded_food.csv`, `food_nutrient.csv`, … |
| Member-file ctimes | 2025-04-06, 18:04–18:05 UTC | `stat` (inode creation on lab storage) |
| Records | 1,981,655, all `data_type = branded_food` | `food.csv` |
| `publication_date` range | 2019-04-01 → 2024-12-19 | `food.csv` |
| Latest publication batches | 2024-09-13, 2024-10-31, 2024-11-14, 2024-12-19 | `food.csv` |

Reading of the timestamps: the mtimes (13 Dec 2024) *precede* the latest record
publication date (19 Dec 2024), so they cannot be the local extraction time — they are
the timestamps carried inside the USDA zip and restored by `unzip`, i.e. when USDA built
the archive. FDC stamps a monthly batch with its scheduled release date, a few days
after the archive is generated. The ctimes (6 Apr 2025) are when the files were written
to this filesystem. The download therefore falls between mid-December 2024 and
6 April 2025; no wget/curl record of it survives in the shell history.

### Where it lands on the download page

FDC's current listing offers Branded Foods only at semi-annual points:
2024-04-18, 2024-10-31, 2025-04-24, 2025-12-18, 2026-04-30 (latest). Record-level
publication dates in the snapshot, however, are monthly (…2024-08-15, 2024-09-13,
2024-10-31, 2024-11-14, 2024-12-19), confirming the interim-release reading. Direct
probes of the archive path confirm the December file is not served:

```
404  https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2024-12-19.zip
404  https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2024-12-13.zip
200  https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2024-10-31.zip
200  https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2025-04-24.zip
```

Divergence from the nearest archived release (2024-10-31), counting only records that
postdate it:

| Table | Rows | Post-2024-10-31 | Share |
|---|---|---|---|
| Raw `food.csv` (branded) | 1,981,655 | 5,014 | 0.25% |
| All-category build | 292,908 | 3,443 | 1.18% |
| Earlier hand-assembled input | 312,252 | 4,291 | 1.37% |

This is a lower bound on the difference: it does not count records the October archive
holds that were superseded or withdrawn by December.

### SR Legacy

| Item | Value | Source |
|---|---|---|
| Release | April 2018, `FoodData_Central_sr_legacy_food_csv_2018-04.zip` | FDC download page |
| Update status | Final release of the Standard Reference data type; not updated | FDC download page |
| Foods in release | 7,793 | USDA (SR Legacy documentation) |
| Local table | `data/SR_legacy_1122.csv`, 7,793 rows (fdc_id, description, nutrient columns) | `wc -l` |
| Local table mtime | 2025-11-23 11:15 UTC (`.CSV` variant 11:12 UTC); filename tag `1122` = 22 Nov | `stat` |

Because SR Legacy is frozen at the April 2018 release, the November 2025 assembly date
carries no version information — any download of SR Legacy, at any date, returns the
same April 2018 data. The row count matching the published 7,793 confirms the local
pool is the complete, unfiltered SR Legacy table.

### Not used

`data/sr28_ndb.xlsx` / `sr28_full.db` (SR28, 2015) are the pre-FDC Standard Reference
release and are separate from SR Legacy; they are not part of the reported pipeline.
