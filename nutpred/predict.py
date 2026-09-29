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
from sklearn.dummy import DummyClassifier
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
DEFAULT_SEARCH_SPACE = {
    "n_estimators_min": 100,
    "n_estimators_max": 1200,
    "max_depth_max": 10,
}

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
    """Occasionally sample exactly 0.0 instead of always a positive log-scaled value."""
    if trial.suggest_categorical(f"{name}_is_zero", [True, False]):
        return 0.0
    return float(trial.suggest_float(name, low, high, log=True))


def _suggest_params(model_type: str, trial: "optuna.Trial", stage: str, search_space: Optional[dict] = None) -> Dict[str, Any]:
    """Sample one trial's hyperparameters for `model_type`/`stage` ('classifier' or 'regressor')."""
    ss = {**DEFAULT_SEARCH_SPACE, **(search_space or {})}
    if model_type == "xgboost":
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", ss["n_estimators_min"], ss["n_estimators_max"], log=True)),
            "max_depth": int(trial.suggest_int("max_depth", 2, int(ss["max_depth_max"]))),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.005, 0.25, log=True)),
            "subsample": float(trial.suggest_float("subsample", 0.55, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.55, 1.0)),
            "min_child_weight": float(trial.suggest_float("min_child_weight", 0.1, 32.0, log=True)),
            "gamma": _optional_zero_log_float(trial, "gamma", 1e-8, 5.0),
            "reg_alpha": _optional_zero_log_float(trial, "reg_alpha", 1e-8, 10.0),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)),
        }
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
    """
    if model_type == "xgboost":
        if stage == "classifier":
            return XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                random_state=random_state, scale_pos_weight=float(scale_pos_weight), n_jobs=1, **params,
            )
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=random_state, n_jobs=1, **params)
    if model_type == "random_forest":
        if stage == "classifier":
            return RandomForestClassifier(random_state=random_state, class_weight="balanced_subsample", n_jobs=1, **params)
        return RandomForestRegressor(random_state=random_state, n_jobs=1, **params)
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
            return _make_scaled_pipeline(KNeighborsClassifier(n_jobs=1, **params))
        return _make_scaled_pipeline(KNeighborsRegressor(n_jobs=1, **params))
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
):
    """Optuna TPE hyperparameter search (k-fold CV per trial via `cv`), then refit
    the best params on the full X/y. `stage` is 'classifier' or 'regressor'."""
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
    if len(y_f) < 2:
        est = _make_estimator(model_type, {}, "regressor", random_state)
        est.fit(X, y_f)
        return est
    n_splits = max(2, min(int(cv), len(y_f)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(model_type, trial, "regressor", search_space)
        trial.set_user_attr("full_params", params)
        scores = []
        for tr, va in splitter.split(X):
            est = _make_estimator(model_type, params, "regressor", random_state)
            est.fit(X[tr], y_f[tr])
            pred = np.maximum(0.0, np.asarray(est.predict(X[va]), dtype=float))
            scores.append(float(np.sqrt(np.mean((y_f[va] - pred) ** 2))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state + 101))
    study.optimize(objective, n_trials=max(1, int(n_trials)), timeout=timeout, show_progress_bar=False)
    best_params = dict(study.best_trial.user_attrs["full_params"])
    best = _make_estimator(model_type, best_params, "regressor", random_state)
    best.fit(X, y_f)
    return best

def _handle_nan_features(X: np.ndarray, feature_set_name: str = "") -> np.ndarray:
    """Helper function to handle NaN values in feature matrices consistently.

    ``keep_empty_features`` matters: without it SimpleImputer *drops* a column
    that is entirely NaN, so the matrix silently narrows here while the
    full-frame predict below still passes every column -- a shape mismatch that
    aborts the whole feature set. An all-NaN column carries no signal either
    way; keeping it as zeros preserves the width.
    """
    if np.any(pd.isna(X)):
        if feature_set_name:
            logger.warning(f"NaN values found in features for {feature_set_name}, using median imputation") 
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X = imputer.fit_transform(X)
    return X

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
    r2 = metric_obj.r2
    rmse = metric_obj.rmse
    mae = metric_obj.mae
    mape = float(mean_absolute_percentage_error(y_true + 1e-6, y_pred))
    s = smape(y_true, y_pred)
    within20 = metric_obj.within20

    out = {
        "Nutrient": nutrient,
        "FeatureSet": feature_set_name,
        "Model": model_label,
        "Group": group_name,
        "SampleType": sample_type,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": s,
        "Within20Pct": within20,
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
    y_bin_tr = (y_tr <= zero_eps).astype(int)
    y_bin_te = (y_te <= zero_eps).astype(int)
    if len(np.unique(y_bin_tr)) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(y_bin_tr[0]))
        clf.fit(Xtr, y_bin_tr)
    else:
        clf = _fit_with_optuna(
            model_type, Xtr, y_bin_tr, cv, "classifier", n_trials, random_state, search_space,
        )
    p_test = _positive_class_proba(clf, Xte)
    if zero_threshold_method in {"youden", "f1"} and len(np.unique(y_bin_te)) < 2:
        threshold = float(zero_threshold)
    else:
        threshold = select_threshold(
            y_bin_te,
            p_test,
            method=zero_threshold_method,
            fixed_threshold=zero_threshold,
        )
    z_pred = (p_test >= threshold).astype(int)
    cls_metrics = evaluate_classifier(
        y_bin_te,
        z_pred,
        p_test,
        target="zero",
        model=f"{model_type}_hurdle_classifier",
    ).as_dict()
    p_full = _positive_class_proba(clf, Xall)
    z_full = (p_full >= threshold).astype(int)

    pos_mask_tr = y_tr > zero_eps
    if not np.any(pos_mask_tr):
        preds_test = np.zeros_like(y_te, dtype=float)
        preds_all = np.zeros(Xall.shape[0], dtype=float)
        reg = None
    else:
        Xtr_pos = Xtr[pos_mask_tr]
        ytr_pos = y_tr[pos_mask_tr].astype(float)
        ytr_target = np.log1p(ytr_pos) if pos_log1p else ytr_pos
        reg = _fit_with_optuna(
            model_type, Xtr_pos, ytr_target, cv, "regressor", n_trials, random_state, search_space,
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
    zero_threshold_method: str = "fixed",
    zero_eps: float = 0.0,
    pos_log1p: bool = False,
    interval_thresholds_per_serving: Optional[Dict[str, float]] = None,
    serving_size_col: Optional[str] = None,
    n_trials: int = 100,
    search_space: Optional[dict] = None,
):
    """
    Train ML models (Optuna-tuned) and evaluate on test set + add predictions
    of all samples. `models_dict[feature_set_name][nutrient]` is always a dict
    with a "regressor" key, plus a "classifier" key when hurdle=True.
    """
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
        "Parameters: test_size=%d, cv=%d, model_type=%s, hurdle=%s",
        len(test_indices),
        cv,
        model_type,
        hurdle,
    )
    
    
    # Check if we have enough samples for train/test split
    if len(food_df) < 10:
        logger.error(f"Not enough samples for ML training: {len(food_df)} samples")
        raise ValueError(f"Not enough samples for ML training: {len(food_df)} samples. Need at least 10 samples.")

    results = []
    models_dict = {}  # Store trained models for SHAP analysis
    
    for nutrient in targets:
        logger.info(f"Training models for nutrient: {nutrient}")
        y = food_df[nutrient]
        y_tr, y_te = y[train_indices].to_numpy(float), y[test_indices].to_numpy(float)
        serving_col = serving_size_col
        if serving_col is None:
            serving_col = "serving_size_g" if "serving_size_g" in food_df.columns else ("serving_size" if "serving_size" in food_df.columns else None)
        serving_te = None if serving_col is None else food_df.loc[test_indices, serving_col].to_numpy(float)
        interval_threshold = None if interval_thresholds_per_serving is None else interval_thresholds_per_serving.get(nutrient)
        
        logger.info(f"Valid training samples for {nutrient}: {len(y_tr)}")

        for feature_set_name, feature_cols in feature_sets.items():
            # Initialize models dict for this feature set if not exists
            if feature_set_name not in models_dict:
                models_dict[feature_set_name] = {}
            logger.info(f"Training ML model with {feature_set_name} features for {nutrient}")
            
            # Check if all feature columns exist
            missing_cols = [col for col in feature_cols if col not in food_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {feature_set_name}: {missing_cols}")
                continue
            
            Xtr = food_df.loc[train_indices, feature_cols].to_numpy(float)
            Xte = food_df.loc[test_indices, feature_cols].to_numpy(float)
            
            # Handle NaN values consistently
            Xtr = _handle_nan_features(Xtr, feature_set_name)
            Xte = _handle_nan_features(Xte, feature_set_name)
           
            logger.debug(f"Feature matrix shapes: Xtr={Xtr.shape}, Xte={Xte.shape}")
            
            try:
                if hurdle:
                    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
                    Xtr_imp = imputer.fit_transform(Xtr)
                    Xte_imp = imputer.transform(Xte)
                    Xall_imp = imputer.transform(food_df[feature_cols].to_numpy(float))
                    preds, full_pred, p_full, z_full, cls_metrics, reg, clf = _train_hurdle(
                        Xtr=Xtr_imp,
                        Xte=Xte_imp,
                        Xall=Xall_imp,
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
                    food_df[f"{nutrient}_pred_fullnut_zero_prob"] = p_full
                    food_df[f"{nutrient}_pred_fullnut_zero_pred"] = z_full
                    pred_col = _fullnut_pred_col_name(nutrient, model_type, group_name, feature_set_name)
                    food_df[pred_col] = full_pred
                else:
                    model = _fit_with_optuna(
                        model_type, Xtr, y_tr, cv, "regressor", n_trials, random_state, search_space,
                    )
                    models_dict[feature_set_name][nutrient] = {"regressor": model}

                    preds = model.predict(Xte)
                    preds = np.maximum(preds, 0)

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
                    metrics_result["Model"] = model_type
                    results.append(metrics_result)

                    preds_all = model.predict(food_df[feature_cols].to_numpy(float))
                    preds_all = np.maximum(preds_all, 0)
                    pred_col = _fullnut_pred_col_name(nutrient, model_type, group_name, feature_set_name)
                    food_df[pred_col] = preds_all
                    logger.info("Added prediction column: %s", pred_col)

            except Exception as e:
                logger.error(f"Error training {feature_set_name} for {nutrient}: {str(e)}")
                continue

    result_df = pd.DataFrame(results)
    id_cols = ["id_col", "SampleType"] + food_df.columns[:3].tolist()
    pred_cols = [col for col in food_df.columns if _is_fullnut_prediction_col(col)]
    _clip_negative_predictions(food_df, pred_cols)
    output_df = food_df[id_cols + pred_cols]
    logger.info(f"ML training complete. Generated {len(result_df)} results (test set and remaining strict sample evaluations)")
    
    return result_df, models_dict, output_df

