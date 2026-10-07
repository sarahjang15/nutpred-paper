import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from nutpred.metrics import evaluate_classifier, evaluate_predictions, select_threshold, smape
import warnings

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

# Default Optuna search-space bounds, overridable via run.py's --config yaml.
# These are the bounds the reported models were tuned with.
DEFAULT_SEARCH_SPACE = {
    "n_estimators_min": 200,
    "n_estimators_max": 5000,
    "max_depth_max": 16,
}

# XGBoost settings used without tuning when a stage has too few rows to
# cross-validate (fewer than MIN_TUNE_ROWS).
MIN_TUNE_ROWS = 20
XGB_UNTUNED_PARAMS = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9,
    "colsample_bytree": 0.9, "min_child_weight": 1.0, "gamma": 0.0,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
}

# Threads per estimator fit. 1 suits small inputs (see _make_estimator); set
# via train_tree_models(n_jobs=...) / run.py --n-jobs for a full-size run --
# the reported runs used 16 XGBoost threads.
N_JOBS = 1

USING_XGB = True
try:
    from xgboost import XGBClassifier, XGBRegressor
    logger.info("XGBoost available for training")
except Exception:
    USING_XGB = False
    XGBClassifier = None
    XGBRegressor = None
    logger.warning("XGBoost not available")

warnings.filterwarnings("ignore", module="xgboost")


def _normalize_model_type(model_type: str) -> str:
    aliases = {
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "svr": "support_vector",
        "svc": "support_vector",
        "support_vector": "support_vector",
        "svm": "support_vector",
        "mlp": "mlp",
        "hist_gradient_boosting": "hist_gradient_boosting",
        "histgb": "hist_gradient_boosting",
        "hgb": "hist_gradient_boosting",
        "linear": "linear",
        "ridge": "linear",
        "knn": "knn",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown model_type: {model_type}")
    norm = aliases[key]
    if norm == "xgboost" and not USING_XGB:
        raise ValueError("model_type='xgboost' requested but xgboost is not installed.")
    return norm


def _positive_class_proba(clf, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return np.zeros(X.shape[0], dtype=float)


def _fullnut_pred_col_name(nutrient: str, model_type: str, group_name: str, feature_set_name: str) -> str:
    return f"{nutrient}_{model_type}_{group_name}_{feature_set_name}"


def _is_fullnut_prediction_col(col: str) -> bool:
    model_tags = (
        "_xgboost_", "_random_forest_", "_support_vector_", "_mlp_",
        "_hist_gradient_boosting_", "_linear_", "_knn_",
    )
    return any(tag in col for tag in model_tags)


def _make_scaled_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("scale", StandardScaler()), ("model", estimator)])


def _optional_zero_log_float(trial: "optuna.Trial", name: str, low: float, high: float) -> float:
    """Occasionally sample exactly 0.0 instead of always a positive log-scaled value.

    Trial parameter names are ``<name>_is_zero`` and ``<name>_pos``, the names
    the reported tuning runs used, so a seeded TPE search proposes the same
    trials.
    """
    if trial.suggest_categorical(f"{name}_is_zero", [True, False]):
        return 0.0
    return float(trial.suggest_float(f"{name}_pos", low, high, log=True))


def _suggest_params(
    model_type: str,
    trial: "optuna.Trial",
    stage: str,
    search_space: Optional[dict] = None,
    allow_tweedie: bool = False,
) -> Dict[str, Any]:
    """Sample one trial's hyperparameters for `model_type`/`stage` ('classifier' or 'regressor').

    ``allow_tweedie`` lets the XGBoost regressor choose a Tweedie objective;
    the reported setup allows it for the one-stage model (which sees the zeros)
    but not for the hurdle model's positive-only regressor.
    """
    ss = {**DEFAULT_SEARCH_SPACE, **(search_space or {})}
    if model_type == "xgboost":
        params: Dict[str, Any] = {}
        if stage == "regressor" and allow_tweedie:
            params["objective"] = trial.suggest_categorical("objective", ["reg:squarederror", "reg:tweedie"])
        params.update({
            "n_estimators": int(trial.suggest_int("n_estimators", ss["n_estimators_min"], ss["n_estimators_max"], log=True)),
            "max_depth": int(trial.suggest_int("max_depth", 3, int(ss["max_depth_max"]))),
            "learning_rate": float(trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)),
            "subsample": float(trial.suggest_float("subsample", 0.5, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.5, 1.0)),
            "min_child_weight": float(trial.suggest_float("min_child_weight", 0.1, 64.0, log=True)),
            "gamma": _optional_zero_log_float(trial, "gamma", 1e-8, 10.0),
            "reg_alpha": _optional_zero_log_float(trial, "reg_alpha", 1e-8, 10.0),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)),
        })
        if params.get("objective") == "reg:tweedie":
            params["tweedie_variance_power"] = float(trial.suggest_float("tweedie_variance_power", 1.1, 1.8))
        return params
    if model_type == "random_forest":
        depth = trial.suggest_categorical("max_depth", [None, 5, 8, 12, 18, 26, 36])
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", 100, 1200, log=True)),
            "max_depth": depth,
            "min_samples_split": int(trial.suggest_int("min_samples_split", 2, 24)),
            "min_samples_leaf": int(trial.suggest_int("min_samples_leaf", 1, 16)),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
            "bootstrap": bool(trial.suggest_categorical("bootstrap", [True, False])),
        }
    if model_type == "hist_gradient_boosting":
        return {
            "max_iter": int(trial.suggest_int("max_iter", 50, 700, log=True)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.005, 0.25, log=True)),
            "max_leaf_nodes": int(trial.suggest_int("max_leaf_nodes", 7, 63)),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": int(trial.suggest_int("min_samples_leaf", 5, 80)),
            "l2_regularization": float(trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)),
        }
    if model_type == "linear":
        if stage == "classifier":
            return {
                "C": float(trial.suggest_float("C", 1e-3, 100.0, log=True)),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }
        return {"alpha": float(trial.suggest_float("alpha", 1e-5, 1e3, log=True))}
    if model_type == "support_vector":
        params: Dict[str, Any] = {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "C": float(trial.suggest_float("C", 1e-2, 100.0, log=True)),
        }
        if params["kernel"] == "rbf":
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        if stage == "classifier":
            params["class_weight"] = "balanced"
            params["probability"] = True
        else:
            params["epsilon"] = float(trial.suggest_float("epsilon", 1e-3, 2.0, log=True))
        return params
    if model_type == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 50, log=True)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        params = {"n_neighbors": n_neighbors, "weights": weights, "metric": metric}
        if metric == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 3)
        return params
    if model_type == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        first_units = trial.suggest_categorical("first_units", [64, 128, 256, 512])
        layers = [first_units]
        for _ in range(1, n_layers):
            layers.append(max(16, layers[-1] // 2))
        return {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": float(trial.suggest_float("alpha", 1e-5, 1e-1, log=True)),
            "learning_rate_init": float(trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True)),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "solver": "adam",
            "max_iter": 2000,
            "learning_rate": "adaptive",
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def _make_estimator(model_type: str, params: Dict[str, Any], stage: str, random_state: int, scale_pos_weight: float = 1.0):
    """Build a fresh, unfitted estimator for `model_type`/`stage` from a sampled params dict.

    n_jobs is pinned to 1 throughout: Optuna fits many small estimators
    sequentially (one per trial/fold), and on many-core machines the
    per-.fit() thread-pool spin-up overhead of n_jobs=-1 dominates runtime
    for these small per-trial data slices (measured ~9s/fit at n_jobs=-1 vs
    ~0.05s/fit at n_jobs=1 on a 128-core box for a tiny 16x8 XGBoost fit).
    On the full cohort, set N_JOBS (run.py --n-jobs) to a fixed thread count.
    """
    if model_type == "xgboost":
        if stage == "classifier":
            return XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                random_state=random_state, scale_pos_weight=float(scale_pos_weight), n_jobs=N_JOBS, **params,
            )
        p = dict(params)
        objective = p.pop("objective", "reg:squarederror")
        return XGBRegressor(objective=objective, tree_method="hist", random_state=random_state, n_jobs=N_JOBS, **p)
    if model_type == "random_forest":
        if stage == "classifier":
            return RandomForestClassifier(random_state=random_state, class_weight="balanced_subsample", n_jobs=N_JOBS, **params)
        return RandomForestRegressor(random_state=random_state, n_jobs=N_JOBS, **params)
    if model_type == "hist_gradient_boosting":
        if stage == "classifier":
            return HistGradientBoostingClassifier(random_state=random_state, **params)
        return HistGradientBoostingRegressor(random_state=random_state, **params)
    if model_type == "linear":
        if stage == "classifier":
            return _make_scaled_pipeline(LogisticRegression(max_iter=5000, solver="lbfgs", random_state=random_state, **params))
        return _make_scaled_pipeline(Ridge(random_state=random_state, **params))
    if model_type == "support_vector":
        if stage == "classifier":
            return _make_scaled_pipeline(SVC(random_state=random_state, cache_size=1000, **params))
        return _make_scaled_pipeline(SVR(cache_size=1000, **params))
    if model_type == "knn":
        if stage == "classifier":
            return _make_scaled_pipeline(KNeighborsClassifier(n_jobs=N_JOBS, **params))
        return _make_scaled_pipeline(KNeighborsRegressor(n_jobs=N_JOBS, **params))
    if model_type == "mlp":
        p = dict(params)
        p["random_state"] = random_state
        if stage == "classifier":
            return _make_scaled_pipeline(MLPClassifier(**p))
        return _make_scaled_pipeline(MLPRegressor(**p))
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_with_optuna(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    stage: str,
    n_trials: int,
    random_state: int,
    search_space: Optional[dict] = None,
    timeout: Optional[int] = None,
    allow_tweedie: bool = False,
    tune_seed: Optional[int] = None,
):
    """Optuna TPE hyperparameter search (k-fold CV per trial via `cv`), then refit
    the best params on the full X/y. `stage` is 'classifier' or 'regressor'.

    ``tune_seed`` seeds the CV folds and the sampler (default ``random_state``);
    the refit estimator always uses ``random_state``. The hurdle model's
    positive-only regressor is tuned with ``random_state + 1``, as in the
    reported runs, so its folds differ from the classifier's.
    """
    tune_seed = random_state if tune_seed is None else int(tune_seed)
    if stage == "classifier":
        y_int = np.asarray(y).astype(int)
        n_pos = int(np.sum(y_int == 1))
        n_neg = int(np.sum(y_int == 0))
        if len(np.unique(y_int)) < 2 or min(n_pos, n_neg) < 2:
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X, y_int)
            return clf
        n_splits = max(2, min(int(cv), n_pos, n_neg))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scale_pos_weight = n_neg / max(n_pos, 1)

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(model_type, trial, "classifier", search_space)
            trial.set_user_attr("full_params", params)
            scores = []
            for tr, va in splitter.split(X, y_int):
                if len(np.unique(y_int[va])) < 2:
                    continue
                est = _make_estimator(model_type, params, "classifier", random_state, scale_pos_weight)
                est.fit(X[tr], y_int[tr])
                prob = _positive_class_proba(est, X[va])
                scores.append(float(roc_auc_score(y_int[va], prob)))
            return float(np.mean(scores)) if scores else 0.5

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=max(1, int(n_trials)), timeout=timeout, show_progress_bar=False)
        best_params = dict(study.best_trial.user_attrs["full_params"])
        best = _make_estimator(model_type, best_params, "classifier", random_state, scale_pos_weight)
        best.fit(X, y_int)
        return best

    # stage == "regressor"
    y_f = np.asarray(y, dtype=float)
    if len(y_f) < MIN_TUNE_ROWS:
        # Too few rows to cross-validate a search; fit fixed settings.
        params = dict(XGB_UNTUNED_PARAMS) if model_type == "xgboost" else {}
        est = _make_estimator(model_type, params, "regressor", random_state)
        est.fit(X, y_f)
        return est
    n_splits = max(2, min(int(cv), len(y_f)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=tune_seed)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(model_type, trial, "regressor", search_space, allow_tweedie=allow_tweedie)
        trial.set_user_attr("full_params", params)
        scores = []
        for tr, va in splitter.split(X):
            est = _make_estimator(model_type, params, "regressor", tune_seed)
            est.fit(X[tr], y_f[tr])
            pred = np.maximum(0.0, np.asarray(est.predict(X[va]), dtype=float))
            scores.append(float(np.sqrt(np.mean((y_f[va] - pred) ** 2))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tune_seed))
    study.optimize(objective, n_trials=max(1, int(n_trials)), timeout=timeout, show_progress_bar=False)
    best_params = dict(study.best_trial.user_attrs["full_params"])
    best = _make_estimator(model_type, best_params, "regressor", random_state)
    best.fit(X, y_f)
    return best


def _clip_negative_predictions(df: pd.DataFrame, cols: List[str]) -> None:
    """Clamp negative prediction values to zero in-place."""
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.where(vals < 0, 0, vals)


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nutrient: str,
    feature_set_name: str,
    group_name: str,
    sample_type: str,
    model_label: str,
    serving_g=None,
    threshold_per_serving: Optional[float] = None,
) -> dict:
    """Helper function to calculate metrics consistently."""
    metric_obj = evaluate_predictions(
        y_true,
        y_pred,
        nutrient,
        model_label,
        serving_g=serving_g,
        threshold_per_serving=threshold_per_serving,
    )
    # MAPE is undefined where the truth is 0, so it is taken over non-zero rows.
    nz = np.asarray(y_true, dtype=float) != 0
    mape = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz])) if nz.any() else float("nan")

    out = {
        "Nutrient": nutrient,
        "FeatureSet": feature_set_name,
        "Model": model_label,
        "Group": group_name,
        "SampleType": sample_type,
        "R2": metric_obj.r2,
        "RMSE": metric_obj.rmse,
        "MAE": metric_obj.mae,
        "MAPE_nonzero": mape,
        "SMAPE": smape(y_true, y_pred),
        "Within20Pct": metric_obj.within20,
        "Within20Pct_inclZero": metric_obj.within20_incl_zero,
    }
    if threshold_per_serving is not None:
        out.update(
            {
                "IntervalAware_RMSE": metric_obj.interval_aware_rmse,
                "IntervalAware_MAE": metric_obj.interval_aware_mae,
                "IntervalAware_Within20Pct": metric_obj.interval_aware_within20,
                "ZeroLabelN": metric_obj.zero_label_n,
                "ZeroLabel_PredZeroOrNearZero": metric_obj.zero_label_pred_zero_or_near_zero,
                "ZeroLabel_PredLowPositiveCompatible": metric_obj.zero_label_pred_low_positive_compatible,
                "ZeroLabel_PredAboveThreshold": metric_obj.zero_label_pred_above_threshold,
            }
        )
    return out


def _train_hurdle(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    Xall: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    model_type: str,
    cv: int,
    random_state: int,
    zero_threshold_method: str,
    zero_threshold: float,
    zero_eps: float,
    pos_log1p: bool,
    n_trials: int = 100,
    search_space: Optional[dict] = None,
):
    """Two-stage model: a zero classifier, then a regressor on positive rows.

    Only training rows choose anything. The classifier is tuned on them, and
    its probability cut-off is picked on the classifier's probabilities for
    those same training rows (select_threshold); test labels are used only to
    report metrics afterwards.
    """
    y_bin_tr = (y_tr <= zero_eps).astype(int)
    y_bin_te = (y_te <= zero_eps).astype(int)
    if len(np.unique(y_bin_tr)) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(y_bin_tr[0]))
        clf.fit(Xtr, y_bin_tr)
        threshold = float(zero_threshold)
    else:
        clf = _fit_with_optuna(
            model_type, Xtr, y_bin_tr, cv, "classifier", n_trials, random_state, search_space,
        )
        threshold = select_threshold(
            y_bin_tr,
            _positive_class_proba(clf, Xtr),
            method=zero_threshold_method,
            fixed_threshold=zero_threshold,
        )
    p_test = _positive_class_proba(clf, Xte)
    z_pred = (p_test >= threshold).astype(int)
    cls_metrics = evaluate_classifier(
        y_bin_te,
        z_pred,
        p_test,
        target="zero",
        model=f"{model_type}_hurdle_classifier",
    ).as_dict()
    cls_metrics["Threshold"] = float(threshold)
    p_full = _positive_class_proba(clf, Xall)
    z_full = (p_full >= threshold).astype(int)

    pos_mask_tr = y_tr > zero_eps
    if not np.any(pos_mask_tr):
        preds_test = np.zeros_like(y_te, dtype=float)
        preds_all = np.zeros(Xall.shape[0], dtype=float)
        reg = None
    elif int(pos_mask_tr.sum()) < MIN_TUNE_ROWS:
        # Too few positive training rows to fit a regressor: predict their mean.
        reg = DummyRegressor(strategy="mean").fit(Xtr[pos_mask_tr], y_tr[pos_mask_tr])
        preds_test = reg.predict(Xte)
        preds_all = reg.predict(Xall)
    else:
        Xtr_pos = Xtr[pos_mask_tr]
        ytr_pos = y_tr[pos_mask_tr].astype(float)
        ytr_target = np.log1p(ytr_pos) if pos_log1p else ytr_pos
        reg = _fit_with_optuna(
            model_type, Xtr_pos, ytr_target, cv, "regressor", n_trials, random_state, search_space,
            tune_seed=random_state + 1,
        )
        preds_test = reg.predict(Xte)
        preds_all = reg.predict(Xall)
        if pos_log1p:
            preds_test = np.expm1(preds_test)
            preds_all = np.expm1(preds_all)
    preds_test = np.where(z_pred == 1, 0.0, preds_test)
    preds_all = np.where(z_full == 1, 0.0, preds_all)
    preds_test = np.maximum(preds_test, 0.0)
    preds_all = np.maximum(preds_all, 0.0)
    return preds_test, preds_all, p_full, z_full, cls_metrics, reg, clf


def train_tree_models(
    food_df: pd.DataFrame,
    feature_sets: dict,
    targets: List[str],
    random_state: int = 42,
    cv: int = 3,
    group_name: str = None,
    model_type: str = "xgboost",
    outdir: str = None,
    hurdle: bool = False,
    force_rf: Optional[bool] = None,
    two_stage: Optional[bool] = None,
    zero_threshold: float = 0.5,
    zero_threshold_method: str = "f1",
    zero_eps: float = 0.0,
    pos_log1p: bool = False,
    interval_thresholds_per_serving: Optional[Dict[str, float]] = None,
    serving_size_col: Optional[str] = None,
    n_trials: int = 100,
    search_space: Optional[dict] = None,
    n_jobs: int = 1,
):
    """
    Train ML models (Optuna-tuned) on ``SampleType == "train"`` rows, score
    them on ``SampleType == "test"`` rows, and add predictions for every row.
    `models_dict[feature_set_name][nutrient]` is always a dict with a
    "regressor" key, plus a "classifier" key when hurdle=True.

    Missing feature values are filled with the TRAIN median (one imputer per
    feature set, fit on train rows only, applied to every row).
    """
    global N_JOBS
    N_JOBS = int(n_jobs)
    logger.info(f"Samples used for ML training: {len(food_df)}")

    test_indices = food_df[food_df.SampleType == 'test'].index.tolist()
    train_indices = food_df[food_df.SampleType == 'train'].index.tolist()

    if force_rf is True:
        model_type = "random_forest"
    if two_stage is not None:
        hurdle = bool(two_stage)
    model_type = _normalize_model_type(model_type)

    logger.info(f"Starting ML training with {len(targets)} targets and {len(feature_sets)} feature sets")
    logger.info(
        "Parameters: train_n=%d, test_n=%d, cv=%d, model_type=%s, hurdle=%s, threshold=%s",
        len(train_indices),
        len(test_indices),
        cv,
        model_type,
        hurdle,
        zero_threshold_method,
    )

    if len(train_indices) < 10 or len(test_indices) < 1:
        raise ValueError(
            f"Not enough samples for ML training: {len(train_indices)} train / "
            f"{len(test_indices)} test rows."
        )

    results = []
    models_dict = {}  # Store trained models for SHAP analysis

    for nutrient in targets:
        logger.info(f"Training models for nutrient: {nutrient}")
        y = food_df[nutrient]
        y_tr, y_te = y[train_indices].to_numpy(float), y[test_indices].to_numpy(float)
        serving_col = serving_size_col
        if serving_col is None:
            serving_col = "serving_size_g" if "serving_size_g" in food_df.columns else ("serving_size" if "serving_size" in food_df.columns else None)
        serving_te = None if serving_col is None else pd.to_numeric(food_df.loc[test_indices, serving_col], errors="coerce").to_numpy(float)
        interval_threshold = None if interval_thresholds_per_serving is None else interval_thresholds_per_serving.get(nutrient)

        logger.info(f"Valid training samples for {nutrient}: {len(y_tr)}")

        for feature_set_name, feature_cols in feature_sets.items():
            if feature_set_name not in models_dict:
                models_dict[feature_set_name] = {}
            logger.info(f"Training ML model with {feature_set_name} features for {nutrient}")

            missing_cols = [col for col in feature_cols if col not in food_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {feature_set_name}: {missing_cols}")
                continue

            X_all_raw = food_df[feature_cols].to_numpy(float)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            imputer.fit(food_df.loc[train_indices, feature_cols].to_numpy(float))
            Xall = imputer.transform(X_all_raw)
            Xtr = imputer.transform(food_df.loc[train_indices, feature_cols].to_numpy(float))
            Xte = imputer.transform(food_df.loc[test_indices, feature_cols].to_numpy(float))
            logger.debug(f"Feature matrix shapes: Xtr={Xtr.shape}, Xte={Xte.shape}")

            pred_col = _fullnut_pred_col_name(nutrient, model_type, group_name, feature_set_name)
            try:
                if hurdle:
                    preds, full_pred, p_full, z_full, cls_metrics, reg, clf = _train_hurdle(
                        Xtr=Xtr,
                        Xte=Xte,
                        Xall=Xall,
                        y_tr=y_tr,
                        y_te=y_te,
                        model_type=model_type,
                        cv=cv,
                        random_state=random_state,
                        zero_threshold_method=zero_threshold_method,
                        zero_threshold=zero_threshold,
                        zero_eps=zero_eps,
                        pos_log1p=pos_log1p,
                        n_trials=n_trials,
                        search_space=search_space,
                    )
                    cls_metrics.update(
                        {
                            "Nutrient": nutrient,
                            "FeatureSet": feature_set_name,
                            "Model": f"{model_type}_hurdle_classifier",
                            "Group": group_name,
                            "SampleType": "test",
                        }
                    )
                    results.append(cls_metrics)
                    models_dict[feature_set_name][nutrient] = {"classifier": clf, "regressor": reg}
                    metric = _calculate_metrics(
                        y_te,
                        preds,
                        nutrient,
                        feature_set_name,
                        group_name,
                        "test",
                        f"{model_type}_hurdle",
                        serving_g=serving_te,
                        threshold_per_serving=interval_threshold,
                    )
                    results.append(metric)
                    # One zero-probability column per feature set (they differ).
                    food_df[f"{pred_col}_zero_prob"] = p_full
                    food_df[f"{pred_col}_zero_pred"] = z_full
                    food_df[pred_col] = full_pred
                else:
                    model = _fit_with_optuna(
                        model_type, Xtr, y_tr, cv, "regressor", n_trials, random_state, search_space,
                        allow_tweedie=True,
                    )
                    models_dict[feature_set_name][nutrient] = {"regressor": model}

                    preds = np.maximum(model.predict(Xte), 0)
                    metrics_result = _calculate_metrics(
                        y_te,
                        preds,
                        nutrient,
                        feature_set_name,
                        group_name,
                        "test",
                        model_type,
                        serving_g=serving_te,
                        threshold_per_serving=interval_threshold,
                    )
                    results.append(metrics_result)
                    food_df[pred_col] = np.maximum(model.predict(Xall), 0)
                logger.info("Added prediction column: %s", pred_col)

            except Exception:
                logger.exception(f"Error training {feature_set_name} for {nutrient}; skipping it")
                continue

    result_df = pd.DataFrame(results)
    id_cols = list(dict.fromkeys(["id_col", "SampleType"] + food_df.columns[:3].tolist()))
    pred_cols = [col for col in food_df.columns if _is_fullnut_prediction_col(col)]
    _clip_negative_predictions(food_df, [c for c in pred_cols if not c.endswith(("_zero_prob", "_zero_pred"))])
    output_df = food_df[id_cols + pred_cols]
    logger.info(f"ML training complete. Generated {len(result_df)} test-set metric rows")

    return result_df, models_dict, output_df
