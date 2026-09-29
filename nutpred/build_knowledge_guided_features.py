"""Knowledge-guided ingredient features (manuscript, Methods: "Knowledge-guided
ingredient features").

Optional preprocessing. These features encode food-composition knowledge that
the data-driven ingredient representations cannot see: reference nutrient
profiles for standardized ingredient terms, the label-order assumption, and
predicted processing yield factors.

The core of it is a constrained ingredient-ratio optimization. For each product
the retained mapped ingredients are matched to analytical reference profiles,
and a per-ingredient ratio vector is fitted so that the reconstructed base
panel approximates the declared one, subject to non-negativity and a monotonic
order constraint (earlier-listed ingredients may not carry less weight than
later ones). The recovered ratios are then combined with the reference profiles
to give an auxiliary prediction per target, denoted ``ing_pred`` in the
manuscript and written here as ``<target>_ing_pred``.

Run it as a pre-step, before run.py::

    python -m nutpred.build_knowledge_guided_features \\
      --food-df data/example/food_df.example.csv \\
      --thesaurus-df data/example/thesaurus.example.xlsx \\
      --ingnut-df data/example/ingnut_df.example.csv \\
      --out-csv data/model_input_kg.csv

run.py then picks the ``*_ing_pred`` columns up automatically and appends them
to every feature set. It does not run the optimizer itself: solving a QP per
product inside the training loop meant re-solving it for every feature set and
every Optuna trial's worth of bookkeeping, for a result that depends only on
the data.

``--method sum`` is the manuscript's baseline for these features: the plain sum
of reference nutrient values over a product's retained mapped ingredients, with
no optimized weights and no yield-factor correction.

Coverage. An auxiliary prediction can only be computed when at least one
retained mapped ingredient has a reference profile. Products without coverage
get NaN, which run.py imputes with the training-set median -- the manuscript's
covered-only analyses restrict to the products where this is not needed.
"""
import re
import numpy as np
import pandas as pd
import cvxpy as cp
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional
import logging
import traceback
import warnings
from nutpred.metrics import r2_manual, smape
from nutpred.preprocess import is_first_mapped
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import nutpred.constants as C

logger = logging.getLogger(__name__)

# Target -> ingnut_df column name. Shared by predict_ingnut_weights_and_targets
# (optimization-based aux) and predict_ingnut_sum_targets (plain-sum aux).
INGNUT_MAPPING = {
    "Energy(kcal)": "Energy",
    "Carbohydrate(g)": "Carbohydrate, by difference",
    "Total fat(g)": "Total lipid (fat)",
    "Protein(g)": "Protein",
    "Sodium(mg)": "Sodium, Na",
    "Total sugar(g)": "Sugars, total",
    "Saturated fatty acids(g)": "Fatty acids, total saturated",
    "Cholesterol(mg)": "Cholesterol",
    "Calcium(mg)": "Calcium, Ca",
    "Fiber(g)": "Fiber, total dietary",
    "Iron(mg)": "Iron, Fe",
}


def _parse_list_cell(value) -> List:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []


# ---------------------------------------------------------------------
# Scaling functions
# ---------------------------------------------------------------------
def make_residual_weights(X_full, Y_mat, mode="std"):
    """
    Returns weights w (length p) so the loss is || diag(w) (X_j b - y_j) ||^2.
    
    Modes: std, pminmax, logiqr
    """
    EPS = 1e-9
    A = np.vstack([X_full.T, Y_mat])  # shape (K+n, p)
    
    if mode == "std":
        s = np.nanstd(A, axis=0, ddof=1)
        w = 1.0 / np.clip(s, EPS, None)
        
    elif mode == "pminmax":
        low = np.nanpercentile(A, 1.0, axis=0)
        high = np.nanpercentile(A, 99.0, axis=0)
        s = high - low
        w = 1.0 / np.clip(s, EPS, None)
        
    elif mode == "logiqr":
        A_pos = np.where(A > 0, A, 0.0)
        A_log = np.log1p(A_pos)
        q1, q3 = np.nanpercentile(A_log, [25.0, 75.0], axis=0)
        s = q3 - q1
        w = 1.0 / np.clip(s, EPS, None)
    else:
        raise ValueError("scale must be one of: std, pminmax, logiqr")
    
    w = np.nan_to_num(w, nan=1.0, posinf=1e6, neginf=1e-6)
    w = np.clip(w, 1e-6, 1e6)
    mu = np.mean(w) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
    w = w / mu
    return w.astype(float)

# ---------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------
def predict_ingnut_weights_and_targets(food_df: pd.DataFrame, ingnut_df: pd.DataFrame, nut8_cols: List[str], 
                                     ingnut_cols: List[str], resolver: str = "rule", constraint: str = "nnls_mono",
                                     ridge: float = 0.0, robust: bool = False, solver_name: str = "osqp", 
                                     scale: str = "std", top_list: List[str] = None, group_name: str = None,
                                     ingredient_col: str = "mapped_list_topk_only", min_weight: float = 0.0,
                                     use_yf: bool = True,
                                     yf_col: Optional[str] = None,
                                     prediction_suffix: str = "_pred_ingnut_yf_aux",
                                     **kwargs) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[int]]:
    """
    Predict ingredient weights and target nutrients using optimization.
    
    Args:
        food_df: DataFrame with food data
        ingnut_df: DataFrame with ingredient-nutrient data
        nut8_cols: List of nutrient column names in food_df
        ingnut_cols: List of nutrient column names in ingnut_df
        resolver: Variant resolution method ("rule" or "first")
        constraint: Optimization constraint type
        ridge: Ridge regularization parameter
        robust: Use robust loss function
        solver_name: Optimization solver name
        scale: Scaling method for residuals
        test_indices: List of test sample indices (if None, use all samples for training)
    
    Returns:
        snack: DataFrame with added prediction columns
        preds_w: Weight matrix (N, K)
        failed: List of failed optimization indices
    """
    logger.info("Starting ingredient-nutrient optimization prediction")
    logger.info(
        "Optimization parameters: resolver=%s, constraint=%s, ridge=%s, robust=%s, solver=%s, scale=%s, min_weight=%s",
        resolver, constraint, ridge, robust, solver_name, scale, min_weight,
    )

    ingnut_mapping = INGNUT_MAPPING

    try:
        # Prepare data matrices using column mapping
        X_all = ingnut_df[[ingnut_mapping[c] for c in nut8_cols]].values # (K, D) nutrient matrix from ingnut data
        Y_mat = food_df[nut8_cols].values        # (N, D) target matrix from snack data
        
        logger.info(f"Data shapes: X_all={X_all.shape}, Y_mat={Y_mat.shape}, K={len(ingnut_df)} ingredients")
        
        # Map ingredients to ingredients
        logger.info("Mapping ingredients to indices...")
        mapped_tokens = [] # list of lists, each list is a list of indices of the ingredients in the top_list

        for j, (_, row) in enumerate(food_df.iterrows()):
            tokens = row.get(ingredient_col, [])
            if not isinstance(tokens, list):
                tokens = []
            
            row_tokens = []
            for token in tokens:
                try:
                    idx = top_list.index(token)
                except ValueError:
                    continue
                row_tokens.append(idx)

            mapped_tokens.append(row_tokens)

        
        logger.info(f"Mapping complete: {len(mapped_tokens)} samples processed")
        
        logger.info("Running optimization on all samples...")
        preds_w, failed = _run_optimization(
            mapped_tokens,
            X_all,
            Y_mat,
            constraint=constraint,
            ridge=ridge,
            robust=robust,
            solver_name=solver_name,
            scale=scale,
            top_list=top_list,
            food_df=food_df,
            min_weight=float(min_weight),
            use_yf=bool(use_yf),
            yf_col=yf_col,
            ingredient_col=ingredient_col,
        )
        
        logger.info(f"Optimization complete. Weight matrix: {preds_w.shape}")
        logger.info(f"Failed optimizations: {len(failed)}")
        
        # Generate predictions for all samples
        logger.info("Generating predictions for all samples...")
        Y_pred = preds_w @ X_all  # (N, D) prediction matrix for nut8 nutrients
        
        # Add prediction columns for nut8 nutrients
        for j, c in enumerate(nut8_cols):
            food_df[f"{c}_opt_{group_name}"] = Y_pred[:, j]
            logger.info(f"Added prediction column: {c}_opt_{group_name}")
        
        # Now use the same weights to predict target nutrients
        # We need to get the target nutrient values from ingnut_df
        logger.info("Generating predictions for target nutrients using same weights...")
        
        for target, ingnut_col in ingnut_mapping.items():
            if ingnut_col in nut8_cols:
                continue
            if ingnut_col in ingnut_df.columns:
                # Get the target nutrient values from ingnut_df for all variants
                target_values = ingnut_df[ingnut_col].values  # (K,) array
                
                # Use the same weights to predict target nutrients
                target_pred = preds_w @ target_values  # (N,) array
                food_df[f"{target}_opt_{group_name}"] = target_pred
                # Backward-compatible alias expected by unified runner
                food_df[f"{target}{prediction_suffix}"] = target_pred
                logger.info(f"Added prediction column: {target}{prediction_suffix} using {ingnut_col}")
            else:
                logger.warning(f"Target nutrient {ingnut_col} not found in ingnut_df, skipping {target}")      
             
        # Add per-row list of weights aligned to mapped_list_topk_only
        try:
            opt_pred_weights = []
            for i in range(len(food_df)):
                tokens = food_df.iloc[i].get(ingredient_col, [])
                if not isinstance(tokens, list):
                    tokens = []
                row_weights = []
                for tok in tokens:
                    try:
                        idx = top_list.index(tok)
                        row_weights.append(float(preds_w[i, idx]))
                    except ValueError:
                        row_weights.append(0.0)
                opt_pred_weights.append(row_weights)
            food_df[f"opt_pred_weights_{group_name}"] = opt_pred_weights
            # Backward-compatible aliases expected by unified runner outputs
            food_df["opt_pred_weights_unified_aux"] = opt_pred_weights
            weight_dicts = []
            for tokens, weights in zip(food_df[ingredient_col], opt_pred_weights):
                d = {}
                if isinstance(tokens, list):
                    for tok, w in zip(tokens, weights):
                        d[str(tok)] = float(w)
                weight_dicts.append(d)
            food_df["opt_pred_weight_dict_unified_aux"] = weight_dicts
            logger.info("Added column with per-food weights aligned to mapped_list_topk_only")
        except Exception as e:
            logger.warning(f"Failed to add opt_pred_weights column: {e}")

        # Return the weights matrix that was used for predictions
        return food_df, preds_w, failed
        
    except Exception as e:
        logger.error(f"Error in predict_ingnut_weights_and_targets: {e}")
        logger.error(traceback.format_exc())
        raise

def _run_optimization(mapped_tokens: List[List[int]], X_all: np.ndarray, Y_mat: np.ndarray,
                     constraint: str = "nnls_mono", ridge: float = 0.0, robust: bool = False,
                     solver_name: str = "osqp", scale: str = "std", top_list: List[str] = None,
                     food_df: pd.DataFrame = None, min_weight: float = 0.0,
                     use_yf: bool = True, yf_col: Optional[str] = None,
                     ingredient_col: str = "mapped_list_topk_only") -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Run optimization for all samples and return weights matrix, variant universe, and failed indices.
    
    constraint: constraint mode for optimization
      - "nnls_only": b >= 0,
      - "nnls_mono": b >= 0, b[k] >= b[k+1]
      - "eq1":       b >= 0, sum(b) == 1
      - "eq1_mono":  b >= 0, sum(b) == 1,    b[k] >= b[k+1]
      - "le1":       b >= 0, sum(b) <= 1
      - "le1_mono":  b >= 0, sum(b) <= 1,    b[k] >= b[k+1]
    
    min_weight:
      If > 0, enforce var >= min_weight on each active ingredient weight (strict positivity floor).
    """
    
    # Choose solver
    solver_map = {"osqp": cp.OSQP, "clarabel": cp.CLARABEL, "scs": cp.SCS, "ecos": cp.ECOS, "piqp": cp.PIQP}
    if solver_name not in solver_map:
        logger.warning(f"Solver {solver_name} not available, falling back to OSQP")
        solver_name = "osqp"
    solver_enum = solver_map[solver_name]
    solver_kwargs = dict(eps_abs=1e-9, eps_rel=1e-9, verbose=False, max_iter=100000, polish=True)
    logger.info(f"Using solver: {solver_name}")
    
    # Prepare optimization matrices
    K = X_all.shape[0]  # Number of variants
    n_snack, p = Y_mat.shape  # Number of samples, number of nutrients
    X_full = X_all.T    # (D, K) - transpose for optimization
    
    
    # Calculate scaling weights
    logger.info(f"Calculating scaling weights using mode: {scale}")
    W_vec = make_residual_weights(X_full, Y_mat, mode=scale)  # length p
    W_mat = np.diag(W_vec)  # (p x p)
    logger.info(f"Scaling weights calculated.")
    
    preds = np.zeros((n_snack, K))
    failed = []
    
    def loss(expr, var):
        """Loss function with optional ridge regularization"""
        base = cp.sum_squares(expr) if not robust else cp.sum(cp.huber(expr, 1.0))
        return base + (ridge * cp.sum_squares(var) if ridge > 0 else 0)
    

    if use_yf and not yf_col:
        raise ValueError(
            "use_yf=True requires yf_col with per-row yield-factor lists aligned to "
            "ingredient_col (e.g. opt_pred_yf_topn after merge_yf_into_snack). "
            "Legacy data/yf_by_food.pkl is no longer supported."
        )
    if use_yf and yf_col:
        logger.info("Using per-row yield factors from column %s", yf_col)
    else:
        logger.info("Optimization without yield-factor scaling (X unscaled)")

    # Solve per snack
    logger.info("Starting optimization for each sample...")

    # jth sample is optimized with the ingredients in the jth list of mapped_tokens
    for j, Sj in enumerate(mapped_tokens):
        if j % 100 == 0:  # Progress update every 100 samples
            logger.info(f"Optimization progress: {j}/{n_snack} samples completed, {len(failed)} failed")

        if not Sj:
            logger.debug(f"Sample {j}: No ingredients to optimize")
            continue

        Kj = len(Sj)
        Xj = X_full[:, Sj]
        yf_j = np.ones(Kj, dtype=float)
        if use_yf and food_df is not None:
            row = food_df.iloc[j]
            if yf_col and yf_col in food_df.columns:
                tokens = row.get(ingredient_col, [])
                if not isinstance(tokens, list):
                    tokens = _parse_list_cell(tokens)
                yf_list = _parse_list_cell(row.get(yf_col, []))
                yf_vals: List[float] = []
                for ti, tok in enumerate(tokens):
                    if tok not in top_list:
                        continue
                    raw = yf_list[ti] if ti < len(yf_list) else 1.0
                    try:
                        yv = float(raw)
                    except (TypeError, ValueError):
                        yv = 1.0
                    if not np.isfinite(yv) or yv <= 0:
                        yv = 1.0
                    yf_vals.append(float(np.clip(yv, 1e-6, 1.0)))
                if len(yf_vals) == Kj:
                    yf_j = np.asarray(yf_vals, dtype=float)
                else:
                    logger.warning(
                        "Sample %s: yf list length %s != ingredient count %s; using ones",
                        j,
                        len(yf_vals),
                        Kj,
                    )
            else:
                logger.warning(
                    "Sample %s: use_yf=True but column %r missing; using unit yield factors",
                    j,
                    yf_col,
                )
        Xj = Xj / yf_j[np.newaxis, :]
        var = cp.Variable(Kj, pos=True) # positive
        
        # Set up constraints based on constraint mode
        cons = []
        #cons.append(var >= 0.001) # positive constraint
        if constraint in ("eq1", "eq1_mono"):
            cons.append(cp.sum(var) == 1)
        elif constraint in ("le1", "le1_mono"):
            cons.append(cp.sum(var) <= 1)
        elif constraint in ("nnls_only", "nnls_mono"):
            pass  # Only non-negativity constraint (handled by nonneg=True)
        else:
            raise ValueError(f"Unknown constraint mode: {constraint}")
        
        # Add monotonicity constraints 
        if constraint.endswith("_mono") and Kj > 1:
            cons += [var[k] >= var[k+1] for k in range(Kj - 1)]

        mw = float(min_weight)
        if mw > 0:
            cons.append(var >= mw)
        
        # Set up residual and solve
        resid = (W_mat @ Xj) @ var - (W_mat @ Y_mat[j])
        prob = cp.Problem(cp.Minimize(loss(resid, var)), cons)
        
        try:
            prob.solve(solver=solver_enum, **solver_kwargs)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(f"Sample {j}: Solver status '{prob.status}', using fallback")
                failed.append(j)
                wj = np.ones(Kj) / Kj
            else: 
                wj = var.value
                #logger.info(f"Sample {j}: Optimization successful with {Kj} ingredients")
        except Exception as e:
            logger.warning(f"Sample {j}: Optimization failed ({str(e)}), using uniform weights")
            failed.append(j)
            wj = np.ones(Kj) / Kj
        
        # Store results
        full = np.zeros(K)
        full[Sj] = wj
        preds[j] = full
    
    logger.info(f"Optimization complete. {len(failed)} samples failed, using fallback weights")
    
    # Use provided top_list for ingredient names, or create generic names if not provided
    if top_list is not None:
        logger.info(f"Using provided top_list for ingredient names: {len(top_list)} ingredients")
    else:
        # Fallback to generic variant names
        raise ValueError(f"top_list not found")
    
    return preds, failed

# ---------------------------------------------------------------------
# Weight-derived features
# ---------------------------------------------------------------------
ALPHA_STAT_SUFFIXES = ("n", "sum", "max", "top3", "hhi", "entropy", "neff", "first", "gt01")


def add_opt_weight_features(
    food_df: pd.DataFrame,
    nut8_cols: List[str],
    group_name: str,
    weights_col: str = "opt_pred_weights_unified_aux",
    prefix: str = "opt",
) -> Tuple[pd.DataFrame, List[str]]:
    """Summarise the optimizer's ingredient weights into fixed-width features.

    ``predict_ingnut_weights_and_targets`` already solves for a per-ingredient
    weight vector and then throws all but its dot-product with the target column
    away. Two blocks recover the rest of that solve:

    **Shape of the weight vector** -- how concentrated the formulation is. A food
    whose mass sits on one ingredient is a different prediction problem from one
    spread over fifteen, and neither ``binary_`` (presence) nor ``score_``
    (rank) can say which it is. Reported as normalised shares so the columns do
    not simply restate ``sum(b)``.

    **Panel reconstruction residual** -- ``(X b)_j`` against the declared ``y_j``,
    per fitted nutrient. This is the optimizer scoring its own fit: weights that
    could not reproduce the panel the solver was given are weights whose implied
    micronutrient content means little, and the model can learn to discount them.

    Only ``nut8_cols`` -- the nutrients that were in the objective -- appear in
    the residual block. A residual over calcium/iron/fiber would be a function of
    the declared target value, so summarising all eleven leaks the targets even
    though the fit itself never saw them.

    Returns the frame and the list of columns added.
    """
    pred_cols = [f"{c}_opt_{group_name}" for c in nut8_cols]
    missing = [c for c in pred_cols if c not in food_df.columns]
    if missing:
        logger.warning(
            "Reconstructed-panel columns absent, skipping residual block: %s", missing
        )

    stat_cols = [f"{prefix}_alpha_{k}" for k in ALPHA_STAT_SUFFIXES]
    stats: Dict[str, List[float]] = {c: [] for c in stat_cols}
    for raw in food_df.get(weights_col, pd.Series([[]] * len(food_df), index=food_df.index)):
        a = np.asarray(
            [v for v in _parse_list_cell(raw) if isinstance(v, (int, float)) and np.isfinite(v)],
            dtype=float,
        )
        a = a[a >= 0]
        if a.size == 0:
            for c in stat_cols:
                stats[c].append(np.nan)
            continue
        total = float(a.sum())
        share = a / total if total > 0 else np.full(a.size, 1.0 / a.size)
        desc = np.sort(share)[::-1]
        hhi = float((share ** 2).sum())
        for c, v in zip(stat_cols, (
            float(a.size), total, float(desc[0]), float(desc[:3].sum()), hhi,
            float(-(share * np.log(np.clip(share, 1e-12, None))).sum()),
            float(1.0 / hhi) if hhi > 0 else np.nan,
            float(share[0]), float((share > 0.01).sum()),
        )):
            stats[c].append(v)
    for c in stat_cols:
        food_df[c] = stats[c]

    res_cols = []
    for nut, pred_col in zip(nut8_cols, pred_cols):
        if pred_col not in food_df.columns or nut not in food_df.columns:
            continue
        declared = pd.to_numeric(food_df[nut], errors="coerce").to_numpy(float)
        predicted = pd.to_numeric(food_df[pred_col], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(declared != 0, (predicted - declared) / np.abs(declared), np.nan)
        col = f"{prefix}_panelres_{nut}"
        food_df[col] = rel
        res_cols.append(col)
    if res_cols:
        # A food whose declared panel is all zeros has no defined residual on any
        # nutrient; nanmedian warns on that all-NaN row rather than erroring.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            food_df[f"{prefix}_panelres_absmed"] = np.nanmedian(
                np.abs(food_df[res_cols].to_numpy(float)), axis=1
            )
        res_cols.append(f"{prefix}_panelres_absmed")

    # A column that is NaN everywhere is not a feature -- on a set where some
    # nutrient is never declared (cholesterol, on most snack rows) its residual
    # is undefined for every food.
    added = stat_cols + res_cols
    degenerate = [c for c in added if food_df[c].isna().all()]
    if degenerate:
        food_df = food_df.drop(columns=degenerate)
        added = [c for c in added if c not in degenerate]
        logger.info("Dropped %d all-NaN optimizer-weight feature(s): %s",
                    len(degenerate), degenerate)
    logger.info("Added %d optimizer-weight features (%d shape, %d residual)",
                len(added),
                len([c for c in added if c in stat_cols]),
                len([c for c in added if c in res_cols]))
    return food_df, added


# ---------------------------------------------------------------------
# Evaluate ing_pred
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# --aux-feature "ingnut_sum": plain sum baseline (no optimization)
# ---------------------------------------------------------------------
def predict_ingnut_sum_targets(
    df: pd.DataFrame,
    ingnut_df: pd.DataFrame,
    ingredient_col: str,
    targets: Optional[List[str]] = None,
    suffix: str = "_pred_ingnut_sum",
) -> pd.DataFrame:
    """Simplest --aux-feature variant: for each target, sum that nutrient's
    ingnut value over the row's mapped ingredients (no optimization/weighting).
    NaN (not 0) when no mapped ingredient has a value for that target -- a row
    with zero matched ingredients should not look identical to a row that
    genuinely sums to zero.
    """
    targets = list(targets or C.TARGET_NUT3)
    name_col = "ing"
    names = ingnut_df[name_col].astype(str).str.upper()
    value_maps: Dict[str, Dict[str, float]] = {}
    for target in targets:
        ingnut_col = INGNUT_MAPPING.get(target)
        if ingnut_col is None or ingnut_col not in ingnut_df.columns:
            logger.warning("predict_ingnut_sum_targets: no ingnut column for %s, skipping", target)
            continue
        vals = pd.to_numeric(ingnut_df[ingnut_col], errors="coerce")
        value_maps[target] = dict(zip(names, vals))

    out = df.copy()
    for target, vmap in value_maps.items():
        preds = np.full(len(out), np.nan, dtype=float)
        n_matched = np.zeros(len(out), dtype=int)
        for j, tokens in enumerate(out[ingredient_col]):
            if not isinstance(tokens, list):
                tokens = _parse_list_cell(tokens)
            total = 0.0
            matched = 0
            for tok in tokens:
                v = vmap.get(str(tok).strip().upper())
                if v is None or not np.isfinite(v):
                    continue
                total += float(v)
                matched += 1
            if matched > 0:
                preds[j] = total
                n_matched[j] = matched
        out[f"{target}{suffix}"] = preds
        out[f"{target}{suffix}_n_matched"] = n_matched
        logger.info(
            "predict_ingnut_sum_targets: %s -> %s/%s rows with >=1 matched ingredient",
            target, int((n_matched > 0).sum()), len(out),
        )
    return out


# ---------------------------------------------------------------------
# Coverage masking: distinguish "no ingredient coverage" from "predicted zero"
# ---------------------------------------------------------------------
def has_ingredient_coverage(
    df: pd.DataFrame, ingredient_col: str, top_list: List[str], min_n: int = 1
) -> pd.Series:
    """True where >= min_n of the row's mapped ingredients are present in top_list.

    Adapted from scripts/yfaux_coverage.py's has_variant_coverage for this
    pipeline's mapped_list columns (that script's version keys off separate
    opt_mapped_list_topn/GPT-pipeline columns which don't exist here).
    """
    top_set = set(top_list)

    def _count(tokens):
        if not isinstance(tokens, list):
            tokens = _parse_list_cell(tokens)
        return sum(1 for t in tokens if t in top_set)

    return df[ingredient_col].apply(_count) >= int(min_n)


def mask_aux_to_missing_if_uncovered(
    df: pd.DataFrame,
    aux_cols: List[str],
    ingredient_col: str,
    top_list: List[str],
    min_n: int = 1,
) -> pd.DataFrame:
    """Set aux_cols to NaN for rows with no covered ingredients, rather than
    letting them keep a misleading computed value of 0 for "nothing matched"."""
    out = df.copy()
    covered = has_ingredient_coverage(out, ingredient_col, top_list, min_n=min_n)
    n_uncovered = int((~covered).sum())
    if n_uncovered:
        logger.info(
            "mask_aux_to_missing_if_uncovered: NaN-ing %s aux column(s) for %s/%s rows with no ingredient coverage",
            len(aux_cols), n_uncovered, len(out),
        )
    for col in aux_cols:
        if col in out.columns:
            out.loc[~covered, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

AUX_SUFFIX = "_ing_pred"


def build_knowledge_guided_features(
    food_df: pd.DataFrame,
    ingnut_df: pd.DataFrame,
    top_list: List[str],
    ingnut_cols: List[str],
    method: str = "opt",
    ingredient_col: str = "mapped_list",
    constraint: str = "nnls_mono",
    solver_name: str = "osqp",
    scale: str = "std",
    ridge: float = 0.0,
    robust: bool = False,
    yf_col: Optional[str] = None,
    weight_features: bool = False,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Add the ``<target>_ing_pred`` auxiliary columns to ``food_df``.

    method
        ``"opt"``     constrained ingredient-ratio optimization (the reported feature)
        ``"yf_opt"``  the same, with the predicted yield-factor correction; needs ``yf_col``
        ``"sum"``     the plain reference-nutrient sum baseline, no optimization

    Returns ``(df, aux_cols, info)``.
    """
    if method not in ("opt", "yf_opt", "sum"):
        raise ValueError(f"method must be opt|yf_opt|sum, got {method!r}")
    if method == "yf_opt" and not yf_col:
        raise ValueError("method='yf_opt' requires yf_col")

    aux_cols = [f"{t}{AUX_SUFFIX}" for t in C.TARGET_NUT3]

    if method == "sum":
        out = predict_ingnut_sum_targets(
            food_df, ingnut_df, ingredient_col=ingredient_col,
            targets=C.TARGET_NUT3, suffix=AUX_SUFFIX,
        )
    else:
        out, _, _ = predict_ingnut_weights_and_targets(
            food_df, ingnut_df, C.NUT8, ingnut_cols,
            ingredient_col=ingredient_col,
            top_list=top_list,
            group_name="full",
            use_yf=(method == "yf_opt"),
            yf_col=yf_col,
            constraint=constraint,
            scale=scale,
            robust=robust,
            solver_name=solver_name,
            prediction_suffix=AUX_SUFFIX,
        )

    out = mask_aux_to_missing_if_uncovered(
        out, aux_cols, ingredient_col=ingredient_col, top_list=top_list,
    )

    if weight_features and method != "sum":
        out, weight_cols = add_opt_weight_features(
            out, C.NUT8, "full", weights_col="opt_pred_weights_full",
        )
        aux_cols = aux_cols + weight_cols
    elif weight_features:
        logger.warning("--weight-features ignored: method='sum' runs no optimizer.")

    covered = int(out[aux_cols[0]].notna().sum()) if aux_cols else 0
    info = {
        "method": method,
        "n_rows": len(out),
        "n_covered": covered,
        "coverage_pct": 100.0 * covered / max(len(out), 1),
        "aux_cols": aux_cols,
    }
    return out, aux_cols, info


def main() -> None:
    import argparse
    import sys
    import time
    from pathlib import Path

    from nutpred.preprocess import (
        ensure_mapped_list_column,
        extract_topk_from_ingnut,
        filter_ingredients_to_topk,
    )

    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--food-df", type=Path,
                    default=root / "data/example/food_df.example.csv")
    ap.add_argument("--thesaurus-df", type=Path,
                    default=root / "data/example/thesaurus.example.xlsx")
    ap.add_argument("--ingnut-df", type=Path,
                    default=root / "data/example/ingnut_df.example.csv")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--method", choices=["opt", "yf_opt", "sum"], default="opt",
                    help="opt (reported), yf_opt (with yield factors), "
                         "sum (reference-nutrient sum baseline)")
    ap.add_argument("--ingredient-cap", type=int, default=20)
    ap.add_argument("--opt-constraint", default="nnls_mono",
                    choices=["nnls_only", "nnls_mono", "eq1", "eq1_mono", "le1", "le1_mono"])
    ap.add_argument("--opt-solver", default="osqp",
                    choices=["osqp", "clarabel", "scs", "ecos", "piqp"])
    ap.add_argument("--opt-scale", default="std", choices=["std", "pminmax", "logiqr"])
    ap.add_argument("--opt-ridge", type=float, default=0.0)
    ap.add_argument("--opt-robust", action="store_true")
    ap.add_argument("--yf-column", default=None,
                    help="column of per-ingredient yield-factor lists; required by --method yf_opt")
    ap.add_argument("--weight-features", action="store_true",
                    help="also emit summary statistics of the fitted weight vector and the "
                         "per-nutrient reconstruction residuals (shared across targets)")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"Loading {args.food_df} ...")
    food = pd.read_csv(args.food_df, low_memory=False)
    thes = pd.read_excel(args.thesaurus_df)
    ingnut = pd.read_csv(args.ingnut_df)
    log(f"  {food.shape[0]} rows | thesaurus {len(thes)} | ingnut {len(ingnut)}")

    cap = args.ingredient_cap if args.ingredient_cap and args.ingredient_cap > 0 else None
    food = ensure_mapped_list_column(food, thes, ingredient_cap=cap)
    # returns (deduplicated ingnut_df, top_list) -- in that order
    ingnut, top_list = extract_topk_from_ingnut(ingnut)
    food = filter_ingredients_to_topk(food, top_list)

    # Positional map from the nut8 label columns onto the ingnut reference
    # column names; same ordering run.py uses.
    ingnut_cols = [INGNUT_MAPPING[c] for c in C.NUT8]

    out, aux_cols, info = build_knowledge_guided_features(
        food, ingnut, top_list, ingnut_cols,
        method=args.method,
        constraint=args.opt_constraint,
        solver_name=args.opt_solver,
        scale=args.opt_scale,
        ridge=args.opt_ridge,
        robust=args.opt_robust,
        yf_col=args.yf_column,
        weight_features=args.weight_features,
    )

    log(f"method={info['method']}  columns: {', '.join(aux_cols[:4])}"
        + (f" (+{len(aux_cols) - 4} more)" if len(aux_cols) > 4 else ""))
    log(f"reference coverage: {info['n_covered']}/{info['n_rows']} "
        f"({info['coverage_pct']:.1f}%) -- uncovered rows are NaN and get "
        f"median-imputed by run.py")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    log(f"Wrote {args.out_csv} ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
