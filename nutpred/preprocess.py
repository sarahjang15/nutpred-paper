import ast
import re
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter
from nutpred.cleaning import clean_ingredient_text
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer 
import nutpred.constants as C

logger = logging.getLogger(__name__)


def load_inputs(food_df_csv: str, thesaurus_xlsx: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load food_df CSV and thesaurus XLSX."""
    food_df = pd.read_csv(food_df_csv)
    thesaurus = pd.read_excel(thesaurus_xlsx)
    return food_df, thesaurus

def make_id_col(food_df: pd.DataFrame) -> pd.DataFrame:
    """Make id column from index"""
    food_df = food_df.copy()
    food_df['id_col'] = food_df.index.astype(str)
    return food_df

def filter_valid_nutrition_data(df):
    """
    Filter out rows with invalid nutrition data based on validation rules.
    
    Returns:
        DataFrame: Filtered DataFrame with only valid rows
    """
    valid_indices = []
    
    for idx, row in df.iterrows():
        is_valid = True
        
        # 1) 에너지가 > 1,080 kcal 초과인 경우
        if pd.notna(row["Energy(kcal)"]) and row["Energy(kcal)"] >= 0:
            if row["Energy(kcal)"] > 1080:
                is_valid = False
                continue
        
        # 2) 에너지가 (탄수화물, 단백질, 지방으로 산출한 에너지) 20% 초과인 경우
        if ((pd.notna(row["Energy(kcal)"]) and row["Energy(kcal)"] >= 0) and 
            (pd.notna(row["Carbohydrate(g)"]) and row["Carbohydrate(g)"] >= 0) and
            (pd.notna(row["Protein(g)"]) and row["Protein(g)"] >= 0) and
            (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0)):
            calculated_energy = (
                (row["Carbohydrate(g)"] if pd.notna(row["Carbohydrate(g)"]) else 0) * 4 +
                (row["Total fat(g)"] if pd.notna(row["Total fat(g)"]) else 0) * 9 +
                (row["Protein(g)"] if pd.notna(row["Protein(g)"]) else 0) * 4
            )
            lower_bound = calculated_energy * 0.8
            upper_bound = calculated_energy * 1.2
            if not (lower_bound <= row["Energy(kcal)"] <= upper_bound):
                is_valid = False
                continue
        
        # 3) 지방, 포화지방, 트랜스지방 이상치 검토
        # 3-1) 포화지방산 > 지방 인 경우
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Saturated fatty acids(g)"]) and row["Saturated fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Saturated fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-2) 트랜스지방산 > 지방 인 경우
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Trans fatty acids(g)"]) and row["Trans fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Trans fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-3) '지방' < ('포화지방산'+'트랜스지방산'+'콜레스테롤/1000') 인 경우
        if pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0:
            trans_fat = row["Trans fatty acids(g)"] if pd.notna(row["Trans fatty acids(g)"]) else 0
            sat_fat = row["Saturated fatty acids(g)"] if pd.notna(row["Saturated fatty acids(g)"]) else 0
            cholesterol = row["Cholesterol(mg)"] / 1000 if pd.notna(row["Cholesterol(mg)"]) else 0
            
            total_fat = trans_fat + sat_fat + cholesterol
            
            if total_fat > row["Total fat(g)"]:
                is_valid = False
                continue
        
        # 4) 탄수화물, 지방, 단백질 이상치 검토: ('100g 당 탄수화물'+ '100g당 단백질'+ '100g당 지방') ≤ 100
        total_macros = sum([
            row["Carbohydrate(g)"] if pd.notna(row["Carbohydrate(g)"]) else 0,
            row["Total fat(g)"] if pd.notna(row["Total fat(g)"]) else 0,
            row["Protein(g)"] if pd.notna(row["Protein(g)"]) else 0
        ])
        
        if total_macros >= 100.0001:
            is_valid = False
            continue
        
        if is_valid:
            valid_indices.append(idx)
        
    df = df.loc[valid_indices]
    df = df[['fdc_id', 'branded_food_category', 'brand_owner', 'market_country', 'description', 'ingredients', 'Energy(kcal)', 'Protein(g)', 'Total fat(g)', 'Carbohydrate(g)', 'Total sugar(g)', 'Saturated fatty acids(g)', 'Cholesterol(mg)', 'Sodium(mg)', 'Fiber(g)', 'Calcium(mg)', 'Iron(mg)']]
    df.reset_index(drop=True, inplace=True).dropna(inplace=True)
    return df


def ensure_mapped_list_column(
    food_df: pd.DataFrame,
    thesaurus: pd.DataFrame,
    ing_col_candidates=("ingredients", "ingredient_list"),
    ingredient_cap: Optional[int] = 20,
) -> pd.DataFrame:
    """
    Create food_df['mapped_list'] using thesaurus mapping over parsed token lists.
    Accepts either a raw 'ingredients' string or a pre-existing 'ingredient_list'.

    ``ingredient_cap`` truncates each label to its first N ingredients before
    mapping; ``None`` or 0 keeps the whole list. The default 20 is the historical
    behaviour and matches replication.constants.MAX_INGREDIENTS, so both
    pipelines cap identically. Uncapping recovers 5.7% of ingredient mentions and
    changes 10.1% of snack foods (1,418 of 14,019 list more than 20).

    Note that the ``*_top20`` column names are kept when uncapped -- they are
    referenced across the codebase -- but they then describe the full list, and
    every filter built on them (``first_mapped``, ``mapped_ratio_top20``,
    ``strict``) is recomputed accordingly, so group membership and N shift.
    """
    # thesaurus mapping (case-insensitive fallbacks)
    possible_t_cols = [
        ("Parsed ingredient term", "Preferred descriptor"),
        ("parsed ingredient term", "preferred descriptor"),
        ("PARSED INGREDIENT TERM", "PREFERRED DESCRIPTOR"),
    ]
    mapping = None
    for a, b in possible_t_cols:
        if a in thesaurus.columns and b in thesaurus.columns:
            mapping = thesaurus.set_index(a)[b].to_dict()
            break
    if mapping is None:
        raise ValueError("Thesaurus must have 'Parsed ingredient term' and 'Preferred descriptor' columns.")

    def normalize_mapping_term(term: str) -> str:
        text = str(term).strip().lower()
        text = re.sub(r"\b(?:select|added)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    normalized_mapping: Dict[str, str] = {}
    for key, value in mapping.items():
        if key is None or (isinstance(key, float) and pd.isna(key)):
            continue
        key_str = normalize_mapping_term(key)
        if not key_str:
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        normalized_mapping[key_str] = str(value)
    mapping = normalized_mapping

    # locate raw ingredient column
    raw_col = next((c for c in ing_col_candidates if c in food_df.columns), None)
    if raw_col is None:
        raise ValueError(f"food_df CSV must contain one of columns: {ing_col_candidates}")

    def to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # allow list-like strings
            try:
                maybe = ast.literal_eval(x)
                if isinstance(maybe, list):
                    return maybe
            except Exception:
                pass
            return clean_ingredient_text(x)
        return []

    def map_term(term: str) -> Tuple[str, bool]:
        raw = str(term).strip()
        candidates = []
        normalized = normalize_mapping_term(raw)
        if normalized:
            candidates.append(normalized)
        raw_lower = raw.lower()
        if raw_lower and raw_lower not in candidates:
            candidates.append(raw_lower)
        for cand in candidates:
            mapped = mapping.get(cand)
            if mapped is not None:
                return str(mapped).upper(), True
        return raw.upper(), False

    def map_terms_all(lst: List[str]) -> List[str]:
        out: List[str] = []
        for term in lst:
            mapped_value, _ = map_term(term)
            out.append(mapped_value)
        return out

    def map_terms_mapped_only(lst: List[str]) -> List[str]:
        out: List[str] = []
        for term in lst:
            mapped_value, was_mapped = map_term(term)
            if was_mapped:
                out.append(mapped_value)
        return out

    def top20_mapped_mask(lst: List[str]) -> List[bool]:
        mask: List[bool] = []
        for term in lst:
            _, was_mapped = map_term(term)
            mask.append(bool(was_mapped))
        return mask

    food_df = food_df.copy()
    food_df["ingredient_list_raw"] = food_df[raw_col].apply(to_list)
    
    # Create ingredient_list_top20 (first `ingredient_cap` items from
    # ingredient_list_raw; the whole list when the cap is None or 0).
    _cap = int(ingredient_cap) if ingredient_cap else 0
    def take_top20(lst):
        if isinstance(lst, list):
            return lst[:_cap] if _cap > 0 else lst
        return []
    
    food_df["ingredient_list_top20"] = food_df["ingredient_list_raw"].apply(take_top20)
    food_df["mapped_list_full"] = food_df["ingredient_list_raw"].apply(map_terms_all)
    food_df["mapped_list"] = food_df["ingredient_list_top20"].apply(map_terms_mapped_only)
    food_df["ingredient_list_top20_mapped_mask"] = food_df["ingredient_list_top20"].apply(top20_mapped_mask)
    food_df["first_item_mapped_thesaurus"] = food_df["ingredient_list_top20_mapped_mask"].apply(
        lambda flags: bool(flags[0]) if isinstance(flags, list) and len(flags) > 0 else False
    )
    food_df["mapped_ratio_top20_thesaurus"] = food_df.apply(
        lambda row: round(
            sum(bool(x) for x in row["ingredient_list_top20_mapped_mask"]) / max(len(row["ingredient_list_top20"]), 1),
            4,
        ),
        axis=1,
    )
    food_df["first_item_mapped_and_ratio70"] = (
        food_df["first_item_mapped_thesaurus"] &
        (food_df["mapped_ratio_top20_thesaurus"] >= 0.7)
    )
    
    # Log ingredient counts
    total_raw_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["ingredient_list_raw"])
    total_top20_ingredients = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["ingredient_list_top20"])
    total_mapped_full = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["mapped_list_full"])
    total_mapped_filtered = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df["mapped_list"])

    logger.info(
        "Ingredient processing: %d raw ingredients -> %d top20 ingredients -> %d mapped (all) -> %d mapped (thesaurus-only)",
        total_raw_ingredients,
        total_top20_ingredients,
        total_mapped_full,
        total_mapped_filtered,
    )
    
    return food_df


def filter_by_ingredients(food_df: pd.DataFrame, include_terms=None) -> pd.DataFrame:
    """Keep only rows whose mapped_list contains any of the included terms."""
    if not include_terms:
        return food_df.reset_index(drop=True)
    def has_included(lst):
        if not isinstance(lst, list):
            return False
        low = [str(x).lower() for x in lst]
        return any(any(term in s for term in include_terms) for s in low)
    mask = food_df["mapped_list"].apply(has_included)
    return food_df.loc[mask].reset_index(drop=True)

def filter_rows(food_df: pd.DataFrame, exclude_terms=None) -> pd.DataFrame:
    """Drop rows whose mapped_list contains any of the excluded terms."""
    if not exclude_terms:
        return food_df.reset_index(drop=True)
    def has_excluded(lst):
        if not isinstance(lst, list):
            return False
        low = [str(x).lower() for x in lst]
        return any(any(term in s for term in exclude_terms) for s in low)
    mask = food_df["mapped_list"].apply(has_excluded)
    return food_df.loc[~mask].reset_index(drop=True)

def make_topk(food_df: pd.DataFrame, k: int) -> List[str]:
    """Return the K most frequent mapped ingredients."""
    freq = Counter()
    for lst in food_df["mapped_list"]:
        if isinstance(lst, list):
            freq.update(lst)
    return [ing for ing, _ in freq.most_common(k)]

def extract_topk_from_ingnut(ingnut_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Extract top-k ingredients from ingnut_df with deduplication, keeping first occurrence."""
    ingredient_col = "ing"
    
    if ingredient_col not in ingnut_df.columns:
        raise ValueError(f"Column '{ingredient_col}' not found in ingnut_df")
    
    full_ingredients = ingnut_df[ingredient_col].tolist()

    # Keep only the first occurrence of each ingredient
    filtered_ingnut_df = ingnut_df.drop_duplicates(subset=[ingredient_col], keep='first')
    unique_ingredients = filtered_ingnut_df[ingredient_col].tolist()
    unique_ingredients = [ing.upper() for ing in unique_ingredients] # Convert to uppercase for case-insensitive matching
    
    # Concise alignment check and example (first 5)
    ing_upper_order = filtered_ingnut_df[ingredient_col].astype(str).str.upper().tolist()
    aligned = (unique_ingredients == ing_upper_order)
    logger.info(f"Top-list alignment with ingnut_df rows: {aligned}")
    logger.info(f"Example top_list vs ingnut rows (first 5): {unique_ingredients[:5]} vs {ing_upper_order[:5]}")
    
    logger.info(f"Extracted {len(unique_ingredients)} unique ingredients from ingnut_df (keeping first occurrence)")
    logger.info(f"Filtered ingnut_df from {len(ingnut_df)} to {len(filtered_ingnut_df)} rows")
    logger.info(f"Removed {len(ingnut_df) - len(filtered_ingnut_df)} duplicate ingredient rows")
    
    return filtered_ingnut_df, unique_ingredients  

def filter_ingredients_to_topk(food_df: pd.DataFrame, top_list: List[str]) -> pd.DataFrame:
    """
    Filter mapped_list to only include ingredients that are in the top-k list from ingnut_df.
    Returns a new column 'mapped_list_topk_only' with filtered ingredients.
    """
    logger.info("Filtering ingredients to top-k list...")

    # Set membership, not list membership: `in top_list` is an O(K) scan, which
    # cost 20 minutes on the full-category run (K=118,771, 2.3M ingredients).
    top_set = set(top_list)

    def filter_to_topk(ingredient_list):
        if not isinstance(ingredient_list, list):
            return []

        filtered_list = []

        for ingredient in ingredient_list:
            ingredient = str(ingredient).upper()
            # Check if the ingredient (or its base name) is in top_list
            if ingredient in top_set:
                filtered_list.append(ingredient)

        return filtered_list
    
    food_df = food_df.copy()
    food_df['mapped_list_topk_only'] = food_df['mapped_list'].apply(filter_to_topk)
    
    # Log statistics
    total_original = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df['mapped_list'])
    total_filtered = sum(len(lst) if isinstance(lst, list) else 0 for lst in food_df['mapped_list_topk_only'])
            
    logger.info(f"Filtered ingredients: {total_original} original -> {total_filtered} in top-k ({total_filtered/max(total_original, 1)*100:.1f}%)")

    return food_df



# Curated ingnut tables do not share one schema: data/ingnut_df_top100.csv ships
# simplified column names, while the top135/top200/_derived lineage uses the
# SR-style names INGNUT_MAPPING expects. Normalize so either can be passed.



# Create first_mapped filter
def is_first_mapped(row):
    # Check if the first ingredient from ingredient_list_top20 is properly mapped and in top-k
    ingredient_list_top20 = row.get('ingredient_list_top20', [])
    mapped_list = row.get('mapped_list', [])
    mapped_topk_list = row.get('mapped_list_topk_only', [])
        
    if not isinstance(ingredient_list_top20, list) or len(ingredient_list_top20) == 0:
        return False
        
    if not isinstance(mapped_list, list) or len(mapped_list) == 0:
        return False
        
    if not isinstance(mapped_topk_list, list) or len(mapped_topk_list) == 0:
        return False
        
    # Get the first ingredient from the original top20 list
    first_raw_ing = ingredient_list_top20[0].strip()
    if first_raw_ing == "":
        return False
        
    # Get the first ingredient from the mapped list (this should be the mapped version of the first raw ingredient)
    first_mapped_ing = mapped_list[0].strip()
    if first_mapped_ing == "":
        return False

    # The first_mapped is TRUE if the first mapped ingredient is present in the top-k list
    # This means it was successfully mapped and survived the top-k filtering
    return first_mapped_ing in mapped_topk_list


def build_binary_and_scores(food_df: pd.DataFrame, top_list: List[str], max_score: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create binary and score features from mapped_list_topk_only (resolved ingredients)
    """
    logger.info("Creating binary and score features from ingredients...")
    
    # Use mapped_list_topk_only instead of mapped_list
    lists = food_df["mapped_list_topk_only"].apply(lambda x: x if isinstance(x, list) else [])
    
    mlb = MultiLabelBinarizer()

    bin_mat = mlb.fit_transform(lists)
    binary_df = pd.DataFrame(bin_mat, columns=[f"binary_{c}" for c in mlb.classes_], index=food_df.index)

    # Align the binary block to top_list, exactly like the score block below.
    # MultiLabelBinarizer only emits columns for ingredients that actually OCCUR, so a
    # top-K list whose members are not all present produced fewer binary than score
    # columns (top-200 table -> 133 score but only 108 binary). Reindexing gives an
    # explicit all-zero column for a never-occurring ingredient, so both blocks are K
    # wide and index-aligned. All-zero columns carry no information and cannot change a
    # tree model's predictions; this only makes the feature count match the top list.
    want = [f"binary_{c}" for c in top_list]
    extra = [c for c in binary_df.columns if c not in set(want)]
    if extra:
        logger.warning(
            "build_binary_and_scores: %d observed ingredient(s) are absent from top_list "
            "and will be dropped from the binary block: %s",
            len(extra), [c[len("binary_"):] for c in extra][:5],
        )
    binary_df = binary_df.reindex(columns=want, fill_value=0)
    
    # Create score features using resolved ingredients (keep the variants for scoring)
    def score_row(lst, top):
        sc = {ing: 0 for ing in top}
        for idx, ing in enumerate(lst[:max_score]):
            if ing in sc:
                sc[ing] = max_score - idx
        return [sc[i] for i in top]
    
    score_mat = [score_row(lst, top_list) for lst in lists]
    score_df = pd.DataFrame(score_mat, columns=[f"score_{c}" for c in top_list], index=food_df.index)
    
    logger.info(f"Created {len(binary_df.columns)} binary features and {len(score_df.columns)} score features")
    return binary_df, score_df

def _expand_umap_list_column(
    food_df: pd.DataFrame, base: str, expected_dim: int
) -> None:
    """
    If `base` (e.g. 'umap_10') exists as a list-like column and no `base_i` numeric
    columns exist yet, expand in place into base_1..base_{expected_dim}.
    """
    if expected_dim <= 0:
        return
    pattern = re.compile(rf"^{re.escape(base)}_\d+$")
    if any(pattern.match(c) for c in food_df.columns):
        return
    if base not in food_df.columns:
        return

    def parse_cell(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, str):
            try:
                arr = ast.literal_eval(v)
                if isinstance(arr, (list, tuple)):
                    return list(arr)
            except Exception:
                pass
            v2 = v.strip().strip("[]")
            if not v2:
                return [0.0] * expected_dim
            try:
                return [float(x) for x in v2.split(",")]
            except Exception:
                return [0.0] * expected_dim
        return [0.0] * expected_dim

    arrs = food_df[base].apply(parse_cell).tolist()
    arrs = [a[:expected_dim] + [0.0] * max(0, expected_dim - len(a)) for a in arrs]
    for i in range(expected_dim):
        food_df[f"{base}_{i+1}"] = [a[i] for a in arrs]


def ensure_umap_columns(food_df: pd.DataFrame, expected_dim: int = 10) -> pd.DataFrame:
    """
    Expand list-like embedding columns when expanded numeric columns are absent:
    umap_2 -> umap_2_1..2, umap_10 -> umap_10_1..N (N defaults to 10), umap_20 -> umap_20_1..20.
    If a base column is missing, that embedding arm is left unchanged (skipped later in run.py).
    """
    _expand_umap_list_column(food_df, "umap_2", 2)
    _expand_umap_list_column(food_df, "umap_10", int(expected_dim))
    _expand_umap_list_column(food_df, "umap_20", 20)
    return food_df

def ensure_base_nutrients(food_df: pd.DataFrame) -> List[str]:
    missing = [c for c in C.NUT8 if c not in food_df.columns]
    if missing:
        raise ValueError(f"Missing base nutrient columns: {missing}")

def ensure_targets(food_df: pd.DataFrame) -> List[str]:
    missing = [c for c in C.TARGET_NUT3 if c not in food_df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

 # Calculate mapped ratio
def calculate_mapped_ratio(row):
    raw_count = len(row.get('ingredient_list_raw', []))
    mapped_count = len(row.get('mapped_list_topk_only', []))
    return round(mapped_count / max(raw_count, 1), 4)

def calculate_mapped_ratio_top20(row):
    raw_count = len(row.get('ingredient_list_top20', []))
    mapped_count = len(row.get('mapped_list_topk_only', []))
    return round(mapped_count / max(raw_count, 1), 4)

def _ingnut_topk_flags(mapped_list: List[str], top_list: List[str], max_n: int = 5) -> Dict[str, bool]:
    if not isinstance(mapped_list, list):
        mapped_list = []
    top_set = set(top_list)
    flags: Dict[str, bool] = {}
    for n in range(1, max_n + 1):
        if len(mapped_list) < n:
            flags[f"ingnut_top{n}_mapped"] = False
            continue
        flags[f"ingnut_top{n}_mapped"] = all(tok in top_set for tok in mapped_list[:n])
    flags["ingnut_fully_mapped"] = bool(mapped_list) and all(tok in top_set for tok in mapped_list)
    return flags


def make_filters(food_df: pd.DataFrame, top_list: List[str]) -> pd.DataFrame:
    """
    Create filter columns based on thesaurus-mapped ingredients and ingnut coverage.
    Note: thesaurus mapping (mapped_list) is distinct from ingnut coverage (top_list).
    """
    first_mapped_mask = food_df.apply(is_first_mapped, axis=1)
    food_df["first_mapped"] = first_mapped_mask
    food_df["mapped_ratio"] = food_df.apply(calculate_mapped_ratio, axis=1)
    food_df["mapped_ratio_top20"] = food_df.apply(calculate_mapped_ratio_top20, axis=1)

    ingnut_flags = food_df["mapped_list"].apply(
        lambda lst: _ingnut_topk_flags(lst, top_list=top_list, max_n=5)
    )
    ingnut_flag_df = pd.DataFrame(list(ingnut_flags), index=food_df.index)
    food_df = pd.concat([food_df, ingnut_flag_df], axis=1)

    filter_combinations = [
        # Filter 1: first_mapped = TRUE
        {"name": "first_mapped", "condition": lambda df: df["first_mapped"] == True},

        # Filter 2: mapped_ratio_top20 >= 0.8
        {"name": "mapped_ratio_top20_high", "condition": lambda df: df["mapped_ratio_top20"] >= 0.8},

        # Filter 3: mapped_ratio >= 0.8
        {"name": "mapped_ratio_high", "condition": lambda df: df["mapped_ratio"] >= 0.8},

        # Filter 4: strict
        {"name": "strict", "condition": lambda df: df["first_mapped"] == True and df["mapped_ratio_top20"] >= 0.8},
    ]

    for filter in filter_combinations:
        food_df[filter["name"]] = food_df.apply(filter["condition"], axis=1)
    return food_df


def drop_outliers_nutrients(
    df: pd.DataFrame,
    columns: Iterable[str],
    method: str = "iqr",
    iqr_multiplier: float = 3.0,
    zscore_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Drop rows with nutrient outliers based on IQR or z-score.
    """
    if method == "none":
        return df
    numeric_df = df.copy()
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in numeric_df.columns:
            continue
        values = pd.to_numeric(numeric_df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if method == "iqr":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask &= values.between(lower, upper) | values.isna()
        elif method == "zscore":
            mean = values.mean()
            std = values.std()
            if std == 0 or pd.isna(std):
                continue
            z = (values - mean) / std
            mask &= (z.abs() <= zscore_threshold) | values.isna()
        else:
            raise ValueError("method must be one of: iqr, zscore, none")
    dropped = int((~mask).sum())
    if dropped:
        logger.info("Dropped %d outlier rows using %s method.", dropped, method)
    return df.loc[mask].reset_index(drop=True)
        
def preprocess_pipeline(
    food_df: pd.DataFrame,
    ingnut_df: pd.DataFrame,
    outlier_method: str = "none",
    outlier_iqr_multiplier: float = 3.0,
    outlier_zscore: float = 4.0,
) -> pd.DataFrame:
    """
    Complete ingredient processing pipeline:
    0. Make id column
    1. Filter to top-k (mapped_list_topk_only)
    2. Create binary/score features from resolved ingredients
    3. Make filters
    4. Join features to food_df dataframe
    """
    logger.info("Preprocessing ingredients...")

    # Step 1: Extract top-k ingredients from ingnut_df
    ingnut_df, top_list = extract_topk_from_ingnut(ingnut_df)

    # Step 2: Filter to top-k
    food_df = filter_ingredients_to_topk(food_df, top_list)

    # Step 3: Make filters
    food_df = make_filters(food_df, top_list=top_list)

    # Step 4: Drop outlier rows (nutrient-based)
    nutrient_cols = [c for c in C.NUT8 + C.TARGET_NUT3 if c in food_df.columns]
    if outlier_method != "none" and nutrient_cols:
        food_df = drop_outliers_nutrients(
            food_df,
            nutrient_cols,
            method=outlier_method,
            iqr_multiplier=outlier_iqr_multiplier,
            zscore_threshold=outlier_zscore,
        )

    # Step 5: Create binary/score features from resolved ingredients
    binary_df, score_df = build_binary_and_scores(food_df, top_list)
    
    # Step 6: Join features to food_df dataframe
    food_df = food_df.join(binary_df).join(score_df)
    
    logger.info("Completed ingredient preprocessing.")
    return food_df
    
__all__ = [
    "load_inputs",
    "make_id_col",
    "filter_valid_nutrition_data",
    "ensure_mapped_list_column",
    "filter_by_ingredients",
    "filter_rows",
    "make_topk",
    "extract_topk_from_ingnut",
    "filter_ingredients_to_topk",
    "is_first_mapped",
    "build_binary_and_scores",
    "ensure_umap_columns",
    "ensure_base_nutrients",
    "ensure_targets",
    "calculate_mapped_ratio",
    "calculate_mapped_ratio_top20",
    "make_filters",
    "drop_outliers_nutrients",
    "preprocess_pipeline",
    "rank_vocabulary",
    "thesaurus_match_report",
    "build_ingredient_feature_blocks",
]


# ---------------------------------------------------------------------------
# Ingredient feature blocks, end to end
# ---------------------------------------------------------------------------
# The three build_* scripts that used to live under scripts/ all ran the same
# four steps -- map through the thesaurus, rank a vocabulary, filter to it,
# emit binary_/score_ blocks -- and differed only in where the vocabulary came
# from and whether the binary block was forced to the full vocabulary width.
# Those differences are parameters, not separate pipelines, so they are one
# function here and one CLI in nutpred/build_ingredient_features.py.


def rank_vocabulary(
    food_df: pd.DataFrame,
    k: int,
    scope: str = "all",
    split_col: str = "data_type",
) -> Tuple[List[str], Dict[str, Any]]:
    """Frequency-rank the top-``k`` mapped ingredients, over all rows or train only.

    ``scope="train"`` ranks over ``split_col == "train"`` rows only. Ranking over
    every row lets test-set ingredient frequency decide which features exist,
    which is the same class of leak that fitting UMAP on all rows would be; the
    train-only scope matches the train-fit/test-transform discipline used for
    the embedding.

    Returns the vocabulary and an audit dict, which reports the overlap between
    the train-only and all-rows vocabularies so the size of the leak can be
    stated rather than assumed.
    """
    if scope not in ("all", "train"):
        raise ValueError(f"scope must be 'all' or 'train', got {scope!r}")

    audit: Dict[str, Any] = {"scope": scope, "k": k, "n_rows": len(food_df)}

    if scope == "train":
        if split_col not in food_df.columns:
            raise KeyError(
                f"scope='train' needs a split column {split_col!r}; the frame has "
                f"none. Pass split_col=, or use scope='all'."
            )
        mask = food_df[split_col].astype(str).str.lower().eq("train")
        if not mask.any():
            raise ValueError(f"no rows with {split_col} == 'train'")
        ranked_on = food_df.loc[mask]
        audit["n_train_rows"] = int(mask.sum())
    else:
        ranked_on = food_df

    top_list = make_topk(ranked_on, k)
    audit["n_vocab"] = len(top_list)

    # Leakage audit: how different would the vocabulary be if test rows counted?
    if scope == "train":
        all_rows = make_topk(food_df, k)
        audit["overlap_with_all_rows"] = len(set(top_list) & set(all_rows))

    # Per-ingredient counts over the rows the ranking actually used.
    freq: Counter = Counter()
    for lst in ranked_on["mapped_list"]:
        if isinstance(lst, list):
            freq.update(lst)
    n = max(len(ranked_on), 1)
    audit["vocab_table"] = pd.DataFrame(
        [
            {
                "rank": i,
                "ingredient": ing,
                "count": freq.get(ing, 0),
                "pct_of_products": 100 * freq.get(ing, 0) / n,
            }
            for i, ing in enumerate(top_list, start=1)
        ]
    )
    return top_list, audit


def thesaurus_match_report(food_df: pd.DataFrame) -> Dict[str, Any]:
    """How much of the parsed ingredient text the thesaurus actually resolved."""
    n_rows = len(food_df)
    parsed_col = "ingredient_list_top20" if "ingredient_list_top20" in food_df.columns else "mapped_list"
    total_parsed = int(sum(len(x) for x in food_df[parsed_col] if isinstance(x, list)))
    total_mapped = int(sum(len(x) for x in food_df["mapped_list"] if isinstance(x, list)))
    rows_with_any = int(sum(1 for x in food_df["mapped_list"] if isinstance(x, list) and len(x) > 0))
    return {
        "n_rows": n_rows,
        "rows_with_mapped": rows_with_any,
        "row_match_pct": 100 * rows_with_any / max(n_rows, 1),
        "tokens_parsed": total_parsed,
        "tokens_mapped": total_mapped,
        "token_match_pct": 100 * total_mapped / max(total_parsed, 1),
    }


def build_ingredient_feature_blocks(
    food_df: pd.DataFrame,
    thesaurus: pd.DataFrame,
    top_k: int = 100,
    top_list: Optional[List[str]] = None,
    vocab_scope: str = "all",
    split_col: str = "data_type",
    ingredient_cap: Optional[int] = 20,
    max_score: int = 20,
    exact_k: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Map ingredients through the thesaurus and build the binary_/score_ blocks.

    Parameters
    ----------
    top_list
        An explicit vocabulary (e.g. the ``ing`` column of the ingnut reference
        table). When given, ``top_k`` and ``vocab_scope`` are ignored and the
        vocabulary is used as supplied.
    vocab_scope
        ``"all"`` or ``"train"``; see :func:`rank_vocabulary`.
    exact_k
        Force the binary block onto the full vocabulary index, so an ingredient
        that never occurs still gets an explicit all-zero column. Without this
        the binary block is only as wide as the *observed* ingredient set while
        the score block is the full vocabulary, and the two come out different
        widths -- which is how a "100 + 100" specification silently became
        108 + 133. Scores cannot simply be sliced to match: score is
        ``max_score - position`` within the filtered list, so changing the
        vocabulary shifts positions and the block must be rebuilt.

    Returns
    -------
    (df, binary_df, score_df, info)
        ``df`` carries ``mapped_list`` and ``mapped_list_topk_only``; ``info``
        holds the vocabulary table, the thesaurus match report and the block
        widths.
    """
    df = ensure_mapped_list_column(food_df.copy(), thesaurus, ingredient_cap=ingredient_cap)
    info: Dict[str, Any] = {"thesaurus_match": thesaurus_match_report(df)}

    if top_list is None:
        top_list, audit = rank_vocabulary(df, top_k, scope=vocab_scope, split_col=split_col)
        info.update(audit)
    else:
        top_list = list(dict.fromkeys(str(x).upper().strip() for x in top_list))
        info.update({"scope": "supplied", "k": len(top_list), "n_vocab": len(top_list)})

    df = filter_ingredients_to_topk(df, top_list)
    binary_df, score_df = build_binary_and_scores(df, top_list, max_score=max_score)
    info["binary_cols_observed"] = int(binary_df.shape[1])

    if exact_k:
        want = [f"binary_{i}" for i in top_list]
        missing = [c for c in want if c not in binary_df.columns]
        binary_df = binary_df.reindex(columns=want, fill_value=0)
        info["binary_cols_filled"] = len(missing)
        info["never_occurring"] = [m[len("binary_"):] for m in missing]

    info["binary_cols"] = int(binary_df.shape[1])
    info["score_cols"] = int(score_df.shape[1])
    info["constant_zero_binary"] = int((binary_df != 0).sum().eq(0).sum())
    info["constant_zero_score"] = int((score_df != 0).sum().eq(0).sum())
    return df, binary_df, score_df, info


# ---------------------------------------------------------------------------
# Resolving ingredient tokens onto the ingnut reference vocabulary
# ---------------------------------------------------------------------------
# Resolve GPT-parsed ingredient descriptors to the ingnut top-N nutrient vocabulary.
#
# Resolution order for one ingredient token:
#
# 1. **Exact match** against the vocab (``top_set``).
# 2. **Thesaurus match**: look the raw token up in
#    ``THESAURUSFORPUBLICRELEASE.XLSX`` ("Parsed ingredient term" -> "Preferred
#    descriptor"); if the resolved descriptor is in the vocab, use it.
# 3. **Hierarchical parent-category fallback**: USDA-style descriptors are a
#    comma-separated hierarchy, general -> specific (e.g.
#    ``OIL, VEGETABLE, PALM OLEIN``). If neither of the above matched, strip
#    trailing comma-qualifiers one at a time and check each shorter prefix
#    against the vocab. A candidate parent is accepted **only if verified
#    empirically**: the specific descriptor's own Ca/Fe/Fiber values (looked up
#    in the fuller ingnut reference tables) must be within ``fallback_tolerance``x
#    of the parent's values (the ones that would actually be substituted in).
#
#    Moisture-changing qualifiers routinely fail this check and are correctly
#    left unmatched rather than guessed, e.g.:
#      - ``CARROT, DRIED`` vs ``CARROT``: Ca 6.4x, Iron 13.1x, Fiber 8.4x off.
#      - ``POTATO FLOUR`` vs ``POTATO``: Ca 7.2x, Iron 2.7x, Fiber 2.5x off.
#    Pure-fat qualifiers are usually safe, e.g.:
#      - ``OIL, VEGETABLE, CANOLA, SOYBEAN`` vs ``OIL, VEGETABLE``: identical.
#    When there's no reference row for the specific descriptor at all (can't
#    verify either way), the token is left unmatched rather than assumed safe.













