import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import nutpred.constants as C


PREPARED_OUTPUT_NUTRIENTS = [
    "Energy(kcal)",
    "Carbohydrate(g)",
    "Total fat(g)",
    "Protein(g)",
    "Sodium(mg)",
    "Total sugar(g)",
    "Saturated fatty acids(g)",
    "Trans fatty acids(g)",
    "Cholesterol(mg)",
    "Calcium(mg)",
    "Fiber(g)",
    "Iron(mg)",
    "Potassium(mg)",
]


def make_full_df_nona(snack_input_new: pd.DataFrame) -> pd.DataFrame:
    """
    Create a non-null dataframe for modeling by dropping rows with NaN on required nutrients.

    This follows the intention of "full_df_nona" in the notebook context, ensuring that
    all NUT11 nutrient columns exist and are non-null.
    """
    required_cols = [c for c in C.NUT11 if c in snack_input_new.columns]
    if not required_cols:
        # If none of the required columns exist yet, return as-is
        return snack_input_new.copy()
    full_df_nona = snack_input_new.dropna(subset=required_cols).reset_index(drop=True)
    return full_df_nona


def filter_valid_nutrition_data(df):
    """
    Filter out rows with invalid nutrition data based on validation rules.
    
    Returns:
        DataFrame: Filtered DataFrame with only valid rows
    """
    valid_indices = []
    
    for idx, row in df.iterrows():
        is_valid = True
        
        # 1) Energy above 1,080 kcal per 100 g
        if pd.notna(row["Energy(kcal)"]) and row["Energy(kcal)"] >= 0:
            if row["Energy(kcal)"] > 1080:
                is_valid = False
                continue
        
        # 2) Declared energy more than 20% away from 4*carb + 9*fat + 4*protein
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
        
        # 3) Fat-component consistency
        # 3-1) Saturated fat > total fat
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Saturated fatty acids(g)"]) and row["Saturated fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Saturated fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-2) Trans fat > total fat
        if (pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0) and (pd.notna(row["Trans fatty acids(g)"]) and row["Trans fatty acids(g)"] >= 0):
            if row["Total fat(g)"] < row["Trans fatty acids(g)"]:
                is_valid = False
                continue
        
        # 3-3) Total fat < saturated + trans + cholesterol/1000
        if pd.notna(row["Total fat(g)"]) and row["Total fat(g)"] >= 0:
            trans_fat = row["Trans fatty acids(g)"] if pd.notna(row["Trans fatty acids(g)"]) else 0
            sat_fat = row["Saturated fatty acids(g)"] if pd.notna(row["Saturated fatty acids(g)"]) else 0
            cholesterol = row["Cholesterol(mg)"] / 1000 if pd.notna(row["Cholesterol(mg)"]) else 0
            
            total_fat = trans_fat + sat_fat + cholesterol
            
            if total_fat > row["Total fat(g)"]:
                is_valid = False
                continue
        
        # 4) Carbohydrate + protein + fat must not exceed 100 g per 100 g
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
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_pipeline(raw_dir: str) -> pd.DataFrame:
    """
    Build a standardized dataframe from FoodData Central raw CSVs.

    Required files in raw_dir:
      - food.csv
      - branded_food.csv
      - food_nutrient.csv
      - nutrient.csv

    Returns a dataframe with columns expected by this project, including:
      ['fdc_id','branded_food_category','brand_owner','market_country','description','ingredients']
      + prepared nutrient columns, including trans fat and potassium
    """
    food_csv = os.path.join(raw_dir, "food.csv")
    branded_csv = os.path.join(raw_dir, "branded_food.csv")
    fn_csv = os.path.join(raw_dir, "food_nutrient.csv")
    nut_csv = os.path.join(raw_dir, "nutrient.csv")

    food = pd.read_csv(food_csv)
    branded = pd.read_csv(branded_csv)
    food_nutrient = pd.read_csv(fn_csv)
    nutrient = pd.read_csv(nut_csv)

    # Deduplicate branded by latest publication_date per gtin_upc, then merge by fdc_id
    branded_for_dedup = branded.copy()
    # Ensure publication_date is available (pull from food if needed)
    if "publication_date" not in branded_for_dedup.columns and "publication_date" in food.columns:
        branded_for_dedup = branded_for_dedup.merge(
            food[["fdc_id", "publication_date"]], on="fdc_id", how="left"
        )
    # Parse and sort by publication_date
    if "publication_date" in branded_for_dedup.columns:
        branded_for_dedup["_pub_dt"] = pd.to_datetime(branded_for_dedup["publication_date"], errors="coerce")
    else:
        # Fallback: use fdc_id ordering if no date
        branded_for_dedup["_pub_dt"] = pd.NaT
    # If gtin_upc exists, sort by publication_date then keep last per gtin_upc
    if "gtin_upc" in branded_for_dedup.columns:
        branded_for_dedup = branded_for_dedup.sort_values(["_pub_dt", "fdc_id"]).drop_duplicates(
            subset=["gtin_upc"], keep="last"
        )
    # Base metadata join (on fdc_id as required)
    base = food.merge(branded_for_dedup, on="fdc_id", how="inner")

    # Select metadata columns if present
    meta_cols = {}
    meta_cols["fdc_id"] = base["fdc_id"]
    meta_cols["description"] = base["description_x"] if "description_x" in base.columns else base.get("description", pd.Series(index=base.index))
    for col in ["branded_food_category", "brand_owner", "market_country", "ingredients"]:
        if col in base.columns:
            meta_cols[col] = base[col]
        else:
            meta_cols[col] = pd.Series([None] * len(base))
    meta_df = pd.DataFrame(meta_cols)

    # Build nutrient wide table
    # Map FDC nutrient names to our standardized column names (no sugar preference; use 'Total Sugars')
    fdc_to_target = {
        "Energy": "Energy(kcal)",
        "Protein": "Protein(g)",
        "Total lipid (fat)": "Total fat(g)",
        "Carbohydrate, by difference": "Carbohydrate(g)",
        "Fatty acids, total saturated": "Saturated fatty acids(g)",
        "Fatty acids, total trans": "Trans fatty acids(g)",
        "Cholesterol": "Cholesterol(mg)",
        "Sodium, Na": "Sodium(mg)",
        "Fiber, total dietary": "Fiber(g)",
        "Calcium, Ca": "Calcium(mg)",
        "Iron, Fe": "Iron(mg)",
        "Total Sugars": "Total sugar(g)",
        "Potassium, K": "Potassium(mg)",
        "Vitamin A, IU": "Vitamin A(IU)",
        "Vitamin A, RAE": "Vitamin A(RAE)",
        "Vitamin C, total ascorbic acid": "Vitamin C(mg)"
    }

    nut_cols = ["id", "name"] + (["unit_name"] if "unit_name" in nutrient.columns else [])
    fn = food_nutrient.merge(nutrient[nut_cols], left_on="nutrient_id", right_on="id", how="left")
    wanted_fdc_names = set(fdc_to_target.keys())
    fn = fn[fn["name"].isin(wanted_fdc_names)].copy()
    # FDC names both energy rows "Energy" (KCAL, id 1008; kJ, id 1062). Keep
    # kcal only, or the de-duplication below picks one of the two arbitrarily.
    is_energy = fn["name"].eq("Energy")
    if "unit_name" in fn.columns:
        fn = fn[~is_energy | fn["unit_name"].astype(str).str.upper().eq("KCAL")]
    else:
        fn = fn[~is_energy | fn["nutrient_id"].eq(1008)]
    fn["target_col"] = fn["name"].apply(lambda n: fdc_to_target.get(n, n))
    # Resolve duplicates (no priority needed when using a single sugar name)
    fn = fn.sort_values(["fdc_id", "target_col"]).drop_duplicates(
        subset=["fdc_id", "target_col"], keep="first"
    )

    wide = fn.pivot(index="fdc_id", columns="target_col", values="amount").reset_index()

    # Join metadata and nutrients
    full = meta_df.merge(wide, on="fdc_id", how="left")

    # Ensure all expected nutrient columns exist (even if missing)
    for col in PREPARED_OUTPUT_NUTRIENTS:
        if col not in full.columns:
            full[col] = np.nan

    # Final ordering
    ordered_cols = [
        "fdc_id", "branded_food_category", "brand_owner", "market_country", "description", "ingredients"
    ] + PREPARED_OUTPUT_NUTRIENTS
    full = full[[c for c in ordered_cols if c in full.columns]]

    # Drop rows that fail the rule-based validity checks
    return filter_valid_nutrition_data(full)


__all__ = [
    "make_full_df_nona",
    "filter_valid_nutrition_data",
    "prepare_pipeline",
    "EMBEDDING_DIMS",
    "DEFAULT_UMAP_DIMS",
    "get_client",
    "embed_texts",
    "embed_texts_cached",
    "fit_umap_list_columns",
    "build_embedding_columns",
]


# ---------------------------------------------------------------------------
# Ingredient-text embeddings, with an on-disk cache
# ---------------------------------------------------------------------------
# The matching and UMAP steps only ever need each DISTINCT ingredient string
# embedded once, however many rows reference it, so embeddings are cached to
# .npz keyed by a hash of the exact text list plus the model identity. That
# matters for cost as much as speed: the embedding call is metered.

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data/embedding_cache"
DEFAULT_MODEL = "text-embedding-3-small"

# Length of the raw vector each OpenAI embedding model returns. A model not
# listed here is probed with one call to learn its length.
EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# UMAP output dimensionalities written as umap_<d> columns. 20 is the reported
# setting; pass dims= (or --embed-dims) for others.
DEFAULT_UMAP_DIMS: Tuple[int, ...] = (20,)

# UMAP settings, as in the reported train-fit / test-transform build.
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05


def get_client(api_key: Optional[str] = None, api_key_file: Optional[str] = None):
    """OpenAI client. The key comes from ``api_key``, else the first line of
    ``api_key_file``, else the OPENAI_API_KEY environment variable. No key is
    stored in this repository; supply your own."""
    from openai import OpenAI  # type: ignore

    key = api_key
    if not key and api_key_file:
        key = Path(api_key_file).expanduser().read_text().strip().splitlines()[0].strip()
    key = key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "OpenAI API key not provided. Pass --api-key-file, or set OPENAI_API_KEY."
        )
    return OpenAI(api_key=key)


def embed_texts(
    texts: List[str],
    client,
    model: str = DEFAULT_MODEL,
    batch_size: int = 256,
    max_retries: int = 5,
) -> np.ndarray:
    """OpenAI embeddings for a list of strings, batched. Already unit-norm on return."""
    dim = EMBEDDING_DIMS.get(model)
    if dim is None:
        # Probe dim from a single call rather than hardcoding an unknown model's size.
        probe = client.embeddings.create(model=model, input=texts[:1] or [" "])
        dim = len(probe.data[0].embedding)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch)
                break
            except Exception as e:  # noqa: BLE001 - rate limits / transient API errors
                wait = 2 ** attempt
                logger.warning("Embedding batch %d failed (%s: %s), retry %d/%d in %ds",
                                start, type(e).__name__, e, attempt + 1, max_retries, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch starting at {start} after {max_retries} retries")
        for i, item in enumerate(response.data):
            out[start + i] = item.embedding
        if start % (batch_size * 4) == 0:
            logger.info("Embedded %d/%d texts", start, len(texts))
    return out


def _cache_key(texts: List[str], backend: str) -> str:
    h = hashlib.sha256()
    h.update(backend.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def embed_texts_cached(
    texts: List[str],
    client,
    model: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 256,
) -> np.ndarray:
    """Same as embed_texts, but memoized to cache_dir/{hash}.npz keyed on (texts, model).

    ``client`` may be a zero-argument factory instead of a client. It is then
    called only on a cache miss, so a fully cached run needs no API key at all
    -- which is the difference between a reproducible re-run and one that
    stops at a credential prompt.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend = f"openai:{model}"
    key = _cache_key(texts, backend)
    cache_path = cache_dir / f"{key}.npz"
    if cache_path.exists():
        logger.info("Loading cached embeddings from %s (%d texts)", cache_path, len(texts))
        data = np.load(cache_path)
        return data["embeddings"]

    logger.info("No cache at %s; computing %d embeddings with backend=%s", cache_path, len(texts), backend)
    if callable(client):
        client = client()
    embeddings = embed_texts(texts, client, model=model, batch_size=batch_size)
    np.savez_compressed(cache_path, embeddings=embeddings, backend=backend)
    logger.info("Cached embeddings to %s", cache_path)
    return embeddings


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


def fit_umap_list_columns(
    X: np.ndarray,
    dims: Iterable[int],
    random_state: int,
    train_mask: Optional[np.ndarray] = None,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
) -> dict[str, np.ndarray]:
    """Reduce ``X`` to each of ``dims`` dimensions with UMAP.

    The reducer is fit on the ``train_mask`` rows only and every other row is
    projected through it with ``transform``, so test rows never shape the
    layout. ``train_mask=None`` fits on all rows (only for frames with no
    held-out rows).
    """
    from umap import UMAP  # type: ignore

    mask = np.ones(len(X), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    if mask.shape[0] != len(X) or not mask.any():
        raise ValueError("train_mask must flag at least one row and match X")
    out: dict[str, np.ndarray] = {}
    for n in dims:
        reducer = UMAP(n_components=int(n), n_neighbors=int(n_neighbors),
                       min_dist=float(min_dist), random_state=int(random_state))
        coords = np.empty((len(X), int(n)), dtype=float)
        coords[mask] = reducer.fit_transform(X[mask])
        if (~mask).any():
            coords[~mask] = reducer.transform(X[~mask])
        out[f"UMAP_{int(n)}"] = coords
    return out


# ---------------------------------------------------------------------------
# Ingredient embedding -> UMAP columns, in one call
# ---------------------------------------------------------------------------

DEFAULT_EMBED_PROMPT = (
    "Ingredients listed in descending order by quantity (from most to least); "
    "parentheses indicate sub-ingredients: "
)


def build_embedding_columns(
    food_df: pd.DataFrame,
    dims: Iterable[int] = DEFAULT_UMAP_DIMS,
    ingredient_col: str = "ingredients",
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    api_key_file: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    random_state: int = 42,
    prompt_prefix: str = DEFAULT_EMBED_PROMPT,
    batch_size: int = 256,
    train_mask: Optional[np.ndarray] = None,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
) -> Tuple[pd.DataFrame, dict]:
    """Embed the ingredient text and reduce it to ``umap_{d}`` list columns.

    The reported model uses ``text-embedding-3-small`` reduced to 20
    dimensions; other ``dims`` exist for the dimensionality ablation.

    Only DISTINCT ingredient strings are sent to the API -- a branded-food
    corpus repeats the same ingredient list across package sizes and flavours,
    so deduplicating first is a large saving on a metered call. The embedding
    step is additionally memoised to ``cache_dir``, so a re-run costs nothing.

    UMAP is fit on the TRAIN rows only and every other row is projected
    through the fitted reducer. The train rows are ``train_mask`` if given,
    otherwise those of :func:`nutpred.split.assign_train_test` with
    ``random_state`` as the seed -- the rows run.py trains on when it is given
    the same table.

    Returns ``(df, info)`` where ``df`` has one list-valued ``umap_{d}`` column
    per requested dimension.
    """
    from nutpred.prepare import embed_texts_cached, get_client  # local alias

    if ingredient_col not in food_df.columns:
        raise KeyError(f"{ingredient_col!r} not in the frame; nothing to embed.")

    df = food_df.copy()
    text = df[ingredient_col].fillna("").astype(str)
    prompts = (prompt_prefix + text).tolist()

    uniq = list(dict.fromkeys(prompts))
    logger.info("Embedding %d distinct ingredient strings (%d rows)", len(uniq), len(prompts))

    # Lazy: only a cache miss needs a key.
    uniq_vecs = embed_texts_cached(uniq, lambda: get_client(api_key, api_key_file), model=model,
                                   cache_dir=cache_dir, batch_size=batch_size)
    lookup = {t: i for i, t in enumerate(uniq)}
    X = np.asarray([uniq_vecs[lookup[t]] for t in prompts], dtype=np.float64)

    if train_mask is None:
        from nutpred.split import TRAIN, assign_train_test
        train_mask = (assign_train_test(df, seed=random_state) == TRAIN).to_numpy()
    train_mask = np.asarray(train_mask, dtype=bool)

    dims = [int(d) for d in dims]
    logger.info("Fitting UMAP to %s on %d train rows, transforming %d others",
                dims, int(train_mask.sum()), int((~train_mask).sum()))
    reduced = fit_umap_list_columns(X, dims, random_state=random_state, train_mask=train_mask,
                                    n_neighbors=n_neighbors, min_dist=min_dist)

    for d in dims:
        arr = reduced[f"UMAP_{d}"]
        # .tolist() gives plain Python floats, so the cell reads "[1.23, ...]".
        # str(list(array)) would write "[np.float64(1.23), ...]" under NumPy 2.
        df[f"umap_{d}"] = [str(np.round(row, 6).tolist()) for row in arr]

    info = {
        "n_rows": len(df),
        "n_distinct_texts": len(uniq),
        "embedding_model": model,
        "embedding_dim": int(X.shape[1]),
        "umap_dims": dims,
        "n_fit_rows": int(train_mask.sum()),
        "n_transformed_rows": int((~train_mask).sum()),
    }
    return df, info
